#include <iostream>
#include <bitset>

using namespace std;


void arithmeticShiftRight(int &A, int &Q, int &Q_minus_1, int bits) {
    int least_significant_bit_Q = Q & 1;
    
    
    Q >>= 1;
    
    Q &= ~(1 << (bits - 1)); 
    Q |= ((A & 1) << (bits - 1));
    
    
    int sign_bit = A & (1 << (bits - 1));
    A >>= 1;
    A &= ~(1 << (bits - 1)); // Clear MSB
    A |= sign_bit; 
    
    Q_minus_1 = least_significant_bit_Q;
}

void boothsAlgorithm(int M, int Q_val, int bits) {
    
    int bit_mask = (1 << bits) - 1;
    
    int A = 0 & bit_mask;
    int Q = Q_val & bit_mask;
    int M_neg = (-M) & bit_mask;
    M = M & bit_mask;
    int Q_minus_1 = 0;
    int count = bits;
    
    cout << "Initial State: \n";
    cout << "A: " << bitset<6>(A) << " | Q: " << bitset<6>(Q) << " | Q_-1: " << Q_minus_1 << "\n\n";
    
    while (count > 0) {
        int Q_0 = Q & 1;
        
        cout << "--- Cycle " << (bits - count + 1) << " --- (Q_0 Q_-1 = " << Q_0 << Q_minus_1 << ")\n";
        
        
        if (Q_0 == 1 && Q_minus_1 == 0) {
            A = (A + M_neg) & bit_mask; // A -> A - M
            cout << "Action: A = A - M -> A: " << bitset<6>(A) << "\n";
        } 
        else if (Q_0 == 0 && Q_minus_1 == 1) {
            A = (A + M) & bit_mask; // A -> A + M
            cout << "Action: A = A + M -> A: " << bitset<6>(A) << "\n";
        }
        
        
        arithmeticShiftRight(A, Q, Q_minus_1, bits);
        cout << "Action: Shift Right -> A: " << bitset<6>(A) << " | Q: " << bitset<6>(Q) << " | Q_-1: " << Q_minus_1 << "\n";
        
        count--;
    }
    
    
    int final_result = (A << bits) | Q;
    
    
    if (final_result & (1 << ((bits * 2) - 1))) {
        final_result |= ~((1 << (bits * 2)) - 1);
    }
    
    cout << "\nFinal Answer in Decimal: " << final_result << "\n";
}

int main() {
    int multiplicand = 5;
    int multiplier = -4;
    int bit_width = 6; 
    
    cout << "Multiplying " << multiplicand << " x " << multiplier << " using Booth's Algorithm\n\n";
    boothsAlgorithm(multiplicand, multiplier, bit_width);
    
    return 0;
}

