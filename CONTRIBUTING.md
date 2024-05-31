# Contributing

Contributions (pull requests) are very welcome! Here's how to get started.

---

**Getting started**

First fork the library on GitHub.

Then clone and install the library in development mode:

```bash
git clone https://github.com/your-username-here/distreqx.git
cd distreqx
pip install -e .
```

Then install the pre-commit hook:

```bash
pip install pre-commit
pre-commit install
```

These hooks use Black and isort to format the code, and flake8 to lint it.

---

**If you're making changes to the code:**

Now make your changes. Make sure to include additional tests if necessary.

If you include a new features, there are 3 required classes of tests:
- Correctness: tests the are against analytic or known solutions that ensure the computation is correct
- Compatibility: tests that check for `jit`, `vmap`, and `grad`-ability of the feature to make sure they behave as expected
- Edge cases: tests that make sure edge cases (e.g. large/small numerics, unexpected dtypes) are either dealt with or fail in an expected manner

Next verify the tests all pass:

```bash
pip install pytest
pytest tests
```

Then push your changes back to your fork of the repository:

```bash
git push
```

Finally, open a pull request on GitHub!

---

**If you're making changes to the documentation:**

Make your changes. You can then build the documentation by doing

```bash
pip install -r docs/requirements.txt
mkdocs serve
```
Then doing `Control-C`, and running:
```
mkdocs serve
```
(So you run `mkdocs serve` twice.)

You can then see your local copy of the documentation by navigating to `localhost:8000` in a web browser.