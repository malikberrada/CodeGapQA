# Publishing this repository on GitHub

Create an empty GitHub repository without automatically adding a README,
licence, or `.gitignore`, then run from the extracted public repository:

```bash
git init
git branch -M main
git add .
git commit -m "Public release 1.0.0"
git remote add origin https://github.com/OWNER/REPOSITORY.git
git push -u origin main
```

Before pushing, verify the release:

```bash
python scripts/verify_public_release.py
git status --short
```

After the first push, create a GitHub release tagged `v1.0.0`. Attach the
repository ZIP only if desired; GitHub automatically creates source archives for
each tag.

Never force-add files ignored by `.gitignore` unless they have been independently
reviewed and sanitized.
