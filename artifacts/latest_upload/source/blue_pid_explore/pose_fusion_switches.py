"""Independent module gates. Defaults change only after paired acceptance."""
import os
# E accepted 2026-11-12: QUICK A->E mean 77.90 -> 86.74 (+8.84), 11 improved / 1 worse,
# Map08+09 10/10 0 -> 7, no map dropped.
ACCEPTED = {'B': False, 'C': False, 'C1': False, 'C2': True, 'D': False, 'E': True,
            'F': False, 'G': False, 'H': True}
def enabled(letter):
    return os.environ.get('FUSION_'+letter, '1' if ACCEPTED[letter] else '0') == '1'


