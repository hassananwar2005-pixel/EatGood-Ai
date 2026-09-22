"""
FastAPI backend for the restaurant order agent (LangGraph).

  GET  /health              -> {"status": "ok"}  (Render uses this)
  GET  /menu                -> dishes, category, how many are in stock
  POST /start               -> new customer session: welcome message + session_id
  POST /chat                -> {"session_id": "...", "message": "2 butter chicken"} -> agent replies
  POST /admin/reset-stock   -> header X-Admin-Key, puts stock back to the MENU numbers

Run:  python -m uvicorn main:app --reload      then open http://localhost:8000/docs
"""
from dotenv import load_dotenv

load_dotenv()  # reads .env on your laptop

import logging
import os
import secrets
import threading
import time
import uuid
from collections import defaultdict, deque

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic import BaseModel, Field, field_validator

import restaurant_order_agent as agent

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("order-agent")

agent.VERBOSE = False   # do not print customers' messages in the server logs
agent.QUIET = True

# One graph for all customers. MemorySaver keeps each customer's state under their session_id.
checkpointer = MemorySaver()
graph = agent.build_graph(checkpointer=checkpointer, web=True)

SESSION_TTL = 2 * 3600      # forget sessions that were idle for 2 hours
START_LIMIT = 10            # new orders per minute per visitor
CHAT_LIMIT = 40             # messages per minute per visitor
ERROR_TEXT = "Sorry, something went wrong on my side. Please type your message again."

app = FastAPI(title="Restaurant Order Agent")

origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=os.getenv("ALLOWED_ORIGIN_REGEX") or None,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------- models
class ChatRequest(BaseModel):
    session_id: str = Field(min_length=8, max_length=64)
    message: str = Field(max_length=500)

    @field_validator("message")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("message cannot be empty")
        return v


class OrderInfo(BaseModel):
    dish_name: str
    required_quantity: int


class ProfileInfo(BaseModel):
    allergies: list[str] = []
    unverified: list[str] = []


class ChatResponse(BaseModel):
    session_id: str
    messages: list[str]               # what the agent says, in order
    steps: list[str] = []             # graph steps that ran (order_confirm, cook, serve, ...) - handy for a progress bar
    quick_replies: list[str] = []     # button suggestions for the chat page
    done: bool = False                # True = the order is finished, start a new one
    final_result: str | None = None   # "COMPLETED" or "NOT COMPLETED" (only when done)
    status: str = ""
    order: OrderInfo | None = None
    user_profile: ProfileInfo = ProfileInfo()
    error: bool = False


# ---------------------------------------------------------------- sessions + rate limit
registry_lock = threading.Lock()
session_locks: dict[str, threading.Lock] = {}
last_seen: dict[str, float] = {}
rate_hits: dict[str, deque] = defaultdict(deque)


def touch(sid: str):
    with registry_lock:
        last_seen[sid] = time.time()
        session_locks.setdefault(sid, threading.Lock())


def forget(sid: str):
    with registry_lock:
        last_seen.pop(sid, None)
        session_locks.pop(sid, None)
    checkpointer.delete_thread(sid)


def purge_old_sessions():
    now = time.time()
    with registry_lock:
        old = [sid for sid, t in last_seen.items() if now - t > SESSION_TTL]
    for sid in old:
        forget(sid)


def client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")        # Render puts the visitor's address here
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_limit(request: Request, bucket: str, limit: int, window: int = 60):
    key = f"{bucket}:{client_ip(request)}"
    now = time.time()
    with registry_lock:
        hits = rate_hits[key]
        while hits and now - hits[0] > window:
            hits.popleft()
        if len(hits) >= limit:
            raise HTTPException(status_code=429, detail="Too many requests. Please slow down a little.")
        hits.append(now)


# ---------------------------------------------------------------- running the graph
def build_response(sid: str, messages: list[str], steps: list[str], error: bool = False) -> ChatResponse:
    snap = graph.get_state({"configurable": {"thread_id": sid}})
    v = snap.values or {}
    done = not snap.next
    od = v.get("order_details") or {}
    profile = v.get("user_profile") or {}
    status = v.get("status", "")

    resp = ChatResponse(
        session_id=sid,
        messages=messages,
        steps=steps,
        quick_replies=["Yes", "No"] if (status == "PARTIAL" and not done) else [],
        done=done,
        final_result=(v.get("final_result") or None) if done else None,
        status=status,
        order=OrderInfo(dish_name=od["dish_name"], required_quantity=od["required_quantity"]) if od.get("dish_name") else None,
        user_profile=ProfileInfo(allergies=profile.get("allergies", []), unverified=profile.get("unverified", [])),
        error=error,
    )
    if done:
        forget(sid)               # order finished: free the memory
    return resp


def run_turn(graph_input, sid: str) -> ChatResponse:
    config = {"configurable": {"thread_id": sid}, "recursion_limit": 100}
    messages: list[str] = []
    steps: list[str] = []
    try:
        for update in graph.stream(graph_input, config, stream_mode="updates"):
            for node, data in update.items():
                if node == "__interrupt__":                 # the graph is now waiting for the customer
                    continue
                steps.append(node)
                if data:
                    messages += [m.content for m in data.get("messages", []) if m.type == "ai"]
    except Exception:
        log.exception("Graph failed for session %s", sid)
        return build_response(sid, messages + [ERROR_TEXT], steps, error=True)   # the user can simply type again
    return build_response(sid, messages, steps)


# ---------------------------------------------------------------- routes
@app.get("/")
def root():
    return {"service": "restaurant-order-agent", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {"status": "ok", "llm": bool(os.getenv("GROQ_API_KEY"))}


@app.get("/menu")
def menu():
    stock = agent._stock_read()["quantities"]
    items = [{"name": dish.title(), "category": cat, "in_stock": stock.get(dish, 0)}
             for cat, dishes in agent.MENU_CATEGORIES.items() for dish in dishes]
    return {"items": items, "restock_in_hours": round(agent.hours_to_restock(), 1)}


@app.post("/start", response_model=ChatResponse)
def start(request: Request):
    rate_limit(request, "start", START_LIMIT)
    purge_old_sessions()
    sid = uuid.uuid4().hex
    touch(sid)
    with session_locks[sid]:
        resp = run_turn(agent.new_state(), sid)
    resp.messages = [agent.WELCOME] + resp.messages
    return resp


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    rate_limit(request, "chat", CHAT_LIMIT)
    sid = req.session_id
    lock = session_locks.get(sid)
    if lock is None or not graph.get_state({"configurable": {"thread_id": sid}}).next:
        raise HTTPException(status_code=404, detail="This order is finished or expired. Please start a new order.")
    touch(sid)
    with lock:                                    # one message at a time per customer
        return run_turn(Command(resume=req.message), sid)


@app.post("/admin/reset-stock")
def reset_stock(x_admin_key: str | None = Header(default=None)):
    admin_key = os.getenv("ADMIN_KEY")
    if not admin_key or not x_admin_key or not secrets.compare_digest(x_admin_key, admin_key):
        raise HTTPException(status_code=403, detail="Not allowed")
    agent.reset_stock()
    return {"status": "stock reset"}