# Contributing

Contributions should preserve the scientific claim boundaries and public-data
sanitization rules.

1. Create a focused branch.
2. Add or update tests.
3. Run `python scripts/verify_public_release.py`.
4. Run the relevant `pytest` tests.
5. Do not commit provider credentials, real job identifiers, account metadata,
   private submission ledgers, or operational recovery history.
6. Describe numerical or scientific changes explicitly in the pull request.

Changes to the registered certificate values or public QASM require a new
versioned release and updated checksums.
