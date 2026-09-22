"""
Test scenarios for restaurant_order_agent.py

Run:  pip install pytest

The tests type the user's answers automatically and force cook/serve to pass or fail,
so the result is the same every time. Every run uses a temporary stock file and a fake clock,
so your real menu_stock.json is never touched. No Groq key is needed (simple rule-based extractor).
"""
import os
import tempfile

os.environ.pop("GROQ_API_KEY", None)   

import restaurant_order_agent as agent

agent.VERBOSE = False
HOUR = 3600
NOW = [1_000_000.0]                    


def use_fresh_stock():
    """Empty temp stock file + fake clock, so every test starts with the full menu."""
    agent.STOCK_FILE = os.path.join(tempfile.mkdtemp(), "menu_stock.json")
    NOW[0] = 1_000_000.0
    agent.clock = lambda: NOW[0]


def run(user_inputs, cook_rolls=(), serve_rolls=(), keep_stock=False):
    """Runs the graph. user_inputs = what the user types, *_rolls = True (success) / False (fail) in order."""
    if not keep_stock:
        use_fresh_stock()
    typed = iter(user_inputs)
    rolls = {"cook": iter(cook_rolls), "serve": iter(serve_rolls)}
    agent.get_input = lambda prompt="": next(typed)
    agent.roll = lambda kind: next(rolls[kind])
    agent.TRACE.clear()
    final = agent.graph.invoke(agent.new_state(), {"recursion_limit": 100})
    return final, list(agent.TRACE)


def ai_messages(final):
    return [m.content for m in final["messages"] if m.type == "ai"]


# ---------------------------------------------------------------- your test cases
def test_tc1_order_retries_exhausted():
    # unrelated question -> partial order rejected -> dish not available -> END (order retries)
    final, trace = run(["What is the capital of France?", "10 mutton rogan josh", "no", "2 sushi"])
    assert final["final_result"] == "NOT COMPLETED"
    assert final["order_retries"] == 0
    assert final["status"] == "NOT_AVAILABLE"
    assert "cook" not in trace and trace[-1] == "finish_failure"


def test_tc2_cook_fails_once_serve_fails_once_then_success():
    final, trace = run(["2 butter chicken"], cook_rolls=[False, True, True], serve_rolls=[False, True])
    assert final["final_result"] == "COMPLETED"
    assert final["status"] == "COMPLETE"
    assert trace.count("cook") == 3 and trace.count("serve") == 2
    assert final["cook_retries"] == 0 and final["serve_retries"] == 1


def test_tc3_serve_fails_twice_cook_retries_exhausted():
    final, trace = run(["10 mutton rogan josh", "no", "2 garlic naan"],
                       cook_rolls=[False, True, True], serve_rolls=[False, False])
    assert final["final_result"] == "NOT COMPLETED"
    assert final["status"] == "SERVE_FAILED"
    assert trace.count("cook") == 3 and trace.count("serve") == 2
    assert final["cook_retries"] == 0                     # cook could not retry again
    assert agent.failure_reason(final) == "cook"
    assert trace[-1] == "finish_failure"


# ---------------------------------------------------------------- order rules
def test_three_unrelated_questions_means_cant_serve():
    final, trace = run(["What is the capital of France?", "tell me a joke", "who are you"])
    assert final["final_result"] == "NOT COMPLETED"
    assert final["order_retries"] == 0
    assert "can't serve you" in ai_messages(final)[-1]
    assert "cook" not in trace


def test_three_not_in_menu_means_cant_serve():
    final, trace = run(["2 sushi", "3 noodles", "1 ramen"])
    assert final["final_result"] == "NOT COMPLETED"
    assert "can't serve you" in ai_messages(final)[-1]


def test_partial_message_asks_shall_we_proceed():
    final, trace = run(["10 mutton rogan josh", "yes"], cook_rolls=[True], serve_rolls=[True])
    msg = [m for m in ai_messages(final) if "we don't have" in m][0]
    assert "we don't have 10 mutton rogan josh" in msg
    assert "We can do 4" in msg and "Shall we proceed" in msg
    assert final["final_result"] == "COMPLETED"
    assert final["order_details"]["required_quantity"] == 4        # went ahead with the 4 available


def test_partial_can_be_accepted_on_last_attempt():
    # two unrelated messages (3 -> 1), partial order uses the last attempt, user still says yes
    final, trace = run(["hello", "tell me a joke", "10 mutton rogan josh", "yes"],
                       cook_rolls=[True], serve_rolls=[True])
    assert final["final_result"] == "COMPLETED"
    assert final["order_details"]["required_quantity"] == 4


def test_short_names_work():
    final, trace = run(["2 biryani"], cook_rolls=[True], serve_rolls=[True])
    assert final["order_details"]["dish_name"] == "chicken biryani"
    final, trace = run(["1 lassi"], cook_rolls=[True], serve_rolls=[True])
    assert final["order_details"]["dish_name"] == "mango lassi"


def test_menu_request_lists_menu_without_using_an_attempt():
    final, trace = run(["what do you have in menu?", "1 lassi"], cook_rolls=[True], serve_rolls=[True])
    assert "Gulab Jamun" in " ".join(ai_messages(final))
    assert final["order_retries"] == 3
    assert final["final_result"] == "COMPLETED"


def test_dish_availability_questions_do_not_use_attempts():
    final, trace = run(["do you have butter chicken?", "do you have sushi?", "1 lassi"],
                       cook_rolls=[True], serve_rolls=[True])
    messages = ai_messages(final)
    assert "Yes, we have Butter Chicken on our menu." in messages
    assert "No, we do not have sushi on our menu." in messages
    assert final["order_retries"] == 3
    assert final["final_result"] == "COMPLETED"


def test_multiple_orders_are_all_processed():
    final, trace = run(["2 gulab jamun, 1 butter chicken, and 3 lachha paratha, 1 lassi"],
                       cook_rolls=[True] * 4, serve_rolls=[True] * 4)
    messages = ai_messages(final)
    assert any("2 x gulab jamun" in message for message in messages)
    assert any("1 x butter chicken" in message for message in messages)
    assert any("3 x laccha paratha" in message for message in messages)
    assert any("1 x mango lassi" in message for message in messages)
    assert trace.count("cook") == 4 and trace.count("serve") == 4
    assert final["pending_orders"] == []
    assert final["final_result"] == "COMPLETED"


def test_natural_multi_order_sentence_with_typo_is_processed():
    text = "i want 2 mutton rogan josh 3 lachha paratha 1 chiccken biryani and 5 samosa 2 lassi"
    final, trace = run([text], cook_rolls=[True] * 5, serve_rolls=[True] * 5)
    messages = ai_messages(final)
    assert any("2 x mutton rogan josh" in message for message in messages)
    assert any("3 x laccha paratha" in message for message in messages)
    assert any("1 x chicken biryani" in message for message in messages)
    assert any("5 x vegetable samosa" in message for message in messages)
    assert any("2 x mango lassi" in message for message in messages)
    assert final["pending_orders"] == []
    assert final["final_result"] == "COMPLETED"


def test_unrelated_question_uses_exact_response():
    final, trace = run(["What is the capital of France?", "1 lassi"],
                       cook_rolls=[True], serve_rolls=[True])
    assert "I can't answer to that question." in ai_messages(final)


# ---------------------------------------------------------------- stock rules
def test_stock_goes_down_only_when_order_completes():
    run(["3 butter chicken"], cook_rolls=[True], serve_rolls=[True])
    assert agent.available_stock("butter chicken") == 7             # completed -> 10 - 3

    run(["3 butter chicken"], cook_rolls=[False, False, False], serve_rolls=[], keep_stock=True)
    assert agent.available_stock("butter chicken") == 7             # cook failed -> no change


def test_stock_is_saved_between_runs():
    run(["3 mutton rogan josh"], cook_rolls=[True], serve_rolls=[True])             # 4 -> 1
    final, trace = run(["2 mutton rogan josh", "no", "2 sushi", "1 ramen"], keep_stock=True)
    partial_msgs = [m for m in ai_messages(final) if "We can do 1" in m]
    assert partial_msgs                                             # second run saw only 1 left


def test_restock_after_12_hours():
    run(["3 butter chicken"], cook_rolls=[True], serve_rolls=[True])
    assert agent.available_stock("butter chicken") == 7
    NOW[0] += 11.9 * HOUR
    assert agent.available_stock("butter chicken") == 7             # not yet
    NOW[0] += 0.2 * HOUR                                            # now 12.1 hours later
    assert agent.available_stock("butter chicken") == 10            # restocked
    assert agent.available_stock("mutton rogan josh") == 4


# ---------------------------------------------------------------- errors + feedback
def test_error_goes_back_to_user():
    # the LLM step fails once -> user is asked again, no order attempt is used
    real_extract = agent.extract
    calls = {"n": 0}

    def flaky(text, awaiting):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("Groq is down")
        return real_extract(text, awaiting)

    agent.extract = flaky
    try:
        final, trace = run(["2 butter chicken", "2 butter chicken"], cook_rolls=[True], serve_rolls=[True])
    finally:
        agent.extract = real_extract
    assert final["final_result"] == "COMPLETED"
    assert final["order_retries"] == 3            # the error did not cost an attempt
    assert final["error_count"] == 0              # reset after it worked again
    assert "error:extract_order" in trace and trace.count("get_user_input") == 2


def test_error_in_cook_goes_back_to_user():
    use_fresh_stock()
    typed = iter(["2 butter chicken", "2 butter chicken"])
    calls = {"n": 0}

    def flaky_roll(kind):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("stove exploded")
        return True

    agent.get_input = lambda prompt="": next(typed)
    agent.roll = flaky_roll
    agent.TRACE.clear()
    final = agent.graph.invoke(agent.new_state(), {"recursion_limit": 100})
    assert final["final_result"] == "COMPLETED"
    assert "error:cook" in agent.TRACE


def test_too_many_errors_closes_the_order():
    real_extract = agent.extract

    def always_fails(text, awaiting):
        raise RuntimeError("Groq is down")

    agent.extract = always_fails
    try:
        final, trace = run(["a", "b", "c"])
    finally:
        agent.extract = real_extract
    assert final["final_result"] == "NOT COMPLETED"
    assert final["status"] == "TOO_MANY_ERRORS"
    assert "technical problems" in ai_messages(final)[-1]


def test_user_gets_feedback_when_cook_or_serve_fails():
    final, trace = run(["2 butter chicken"], cook_rolls=[False, True, True], serve_rolls=[False, True])
    msgs = ai_messages(final)
    assert any("cooking your butter chicken did not go well" in m for m in msgs)   # cook failed once
    assert any("could not serve your butter chicken" in m for m in msgs)           # serve failed once
    assert "complete" in msgs[-1]


def test_no_false_hope_when_retries_are_over():
    # TC3: 1 cook-failed + 1 serve-failed feedback, then the final serve failure gives only the apology
    final, trace = run(["2 garlic naan"], cook_rolls=[False, True, True], serve_rolls=[False, False])
    msgs = ai_messages(final)
    feedback = [m for m in msgs if "cooking it again" in m]
    assert len(feedback) == 2
    assert "not completed" in msgs[-1] and "cooking it again" not in msgs[-1]


# ---------------------------------------------------------------- allergy: user_profile + safe recommendations
PEANUT_DISHES = ["Vegetable Samosa", "Palak Patta Chaat"]          # tagged 'peanut' in ALLERGENS


def recommendation_msgs(final):
    return [m for m in ai_messages(final) if "dishes I can suggest" in m or "suggest any dish" in m
            or "safely recommend" in m]


def test_allergy_is_saved_in_user_profile():
    final, trace = run(["I'm highly allergic to peanuts", "2 butter chicken"], cook_rolls=[True], serve_rolls=[True])
    assert final["user_profile"]["allergies"] == ["peanut"]
    assert "highly allergic to peanuts" in final["user_profile"]["notes"][0]
    assert final["order_retries"] == 3            # telling us about an allergy does not cost an attempt
    assert final["final_result"] == "COMPLETED"


def test_recommendation_is_checked_against_the_allergy():
    final, trace = run(["I'm highly allergic to peanuts", "what do you recommend?", "2 butter chicken"],
                       cook_rolls=[True], serve_rolls=[True])
    msg = recommendation_msgs(final)[0]
    assert not any(d in msg for d in PEANUT_DISHES)
    assert "Paneer Tikka" in msg and "confirm with our staff" in msg


def test_whole_menu_check_no_recommended_dish_has_the_allergen():
    use_fresh_stock()
    no_allergy = {"allergies": [], "unverified": [], "notes": []}
    peanut = {"allergies": ["peanut"], "unverified": [], "notes": []}
    everything, _ = agent.safe_recommendations(no_allergy, limit=100)
    safe, _ = agent.safe_recommendations(peanut, limit=100)
    assert "vegetable samosa" in everything or "palak patta chaat" in everything or len(everything) < 15
    # every dish the agent may recommend to a peanut-allergic customer has no peanut tag
    for category_dishes in agent.MENU_CATEGORIES.values():
        for dish in category_dishes:
            picks, _ = agent.safe_recommendations(peanut, limit=1, prefer_dish=dish)
            assert all("peanut" not in agent.ALLERGENS[d] for d in picks)
    assert all("peanut" not in agent.ALLERGENS[d] for d in safe)


def test_allergy_is_remembered_for_the_rest_of_the_conversation():
    final, trace = run(["hello", "I'm allergic to peanuts", "tell me a joke", "what should I eat?", "2 butter chicken"],
                       cook_rolls=[True], serve_rolls=[True])
    msg = recommendation_msgs(final)[0]
    assert not any(d in msg for d in PEANUT_DISHES)
    assert final["user_profile"]["allergies"] == ["peanut"]


def test_alternatives_after_not_available_respect_the_allergy():
    # samosa sold out -> agent suggests other starters. With a peanut allergy palak patta chaat must not appear.
    use_fresh_stock()
    agent.take_stock("vegetable samosa", 20)
    with_allergy, _ = run(["I'm allergic to peanuts", "2 vegetable samosa", "2 sushi", "1 ramen"], keep_stock=True)
    no_allergy, _ = run(["2 vegetable samosa", "2 sushi", "1 ramen"], keep_stock=True)
    msg_allergy = [m for m in ai_messages(with_allergy) if "is not available" in m][0]
    msg_plain = [m for m in ai_messages(no_allergy) if "is not available" in m][0]
    assert "Palak Patta Chaat" not in msg_allergy and "Paneer Tikka" in msg_allergy
    assert "Palak Patta Chaat" in msg_plain                          # without the allergy it is offered


def test_unknown_allergy_means_no_recommendations():
    final, trace = run(["I'm allergic to kiwi", "recommend something", "2 butter chicken"],
                       cook_rolls=[True], serve_rolls=[True])
    assert final["user_profile"]["unverified"] == ["kiwi"]
    msg = recommendation_msgs(final)[0]
    assert "can't safely recommend" in msg and "Butter Chicken" not in msg


def test_dish_without_allergen_info_is_never_recommended_to_an_allergic_customer():
    use_fresh_stock()
    original = dict(agent.ALLERGENS)
    del agent.ALLERGENS["jeera rice"]
    try:
        peanut = {"allergies": ["peanut"], "unverified": [], "notes": []}
        none = {"allergies": [], "unverified": [], "notes": []}
        assert "jeera rice" not in agent.safe_recommendations(peanut, limit=100, prefer_dish="jeera rice")[0]
        assert "jeera rice" in agent.safe_recommendations(none, limit=100, prefer_dish="jeera rice")[0]
    finally:
        agent.ALLERGENS.clear()
        agent.ALLERGENS.update(original)


def test_allergy_and_order_in_one_message():
    final, trace = run(["2 butter chicken, I'm allergic to peanuts"], cook_rolls=[True], serve_rolls=[True])
    assert final["user_profile"]["allergies"] == ["peanut"]
    assert final["order_details"]["dish_name"] == "butter chicken"
    assert final["final_result"] == "COMPLETED"


def test_allergy_words_are_understood():
    use_fresh_stock()
    assert agent.extract_with_rules("I have a peanut allergy").allergies == ["peanut"]
    assert agent.canonical_allergens(["peanuts"]) == (["peanut"], [])
    assert agent.canonical_allergens(["nuts"]) == (["peanut", "tree_nut"], [])       # 'nuts' -> avoid both
    assert agent.canonical_allergens(["Milk", "eggs"]) == (["dairy", "egg"], [])
    assert agent.canonical_allergens(["kiwi"]) == ([], ["kiwi"])


def test_allergy_during_partial_question_does_not_lose_the_order():
    final, trace = run(["10 mutton rogan josh", "I'm allergic to peanuts", "yes"], cook_rolls=[True], serve_rolls=[True])
    assert any("Back to your order" in m for m in ai_messages(final))
    assert final["order_retries"] == 2               # only the partial answer cost an attempt
    assert final["order_details"]["required_quantity"] == 4 and final["final_result"] == "COMPLETED"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as e:
                print(f"FAIL  {name}  {e!r}")
