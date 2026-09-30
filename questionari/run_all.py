"""Esegue tutti gli script dei questionari in sequenza.

    python run_all.py [--risposte DIR] [--out DIR]
"""

import sys

import demografico, edinburgh, nasa, sus

if __name__ == "__main__":
    for mod in (demografico, nasa, edinburgh, sus):
        print(f"\n===== {mod.__name__} =====")
        try:
            mod.main()
        except FileNotFoundError as e:
            print(f"SALTATO: {e}", file=sys.stderr)
