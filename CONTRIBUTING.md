# Contributing guidelines

👋 Contributions are welcome!

Once you've tested your changes and verified that they work, you can commit your changes but please setup a git hook to run clippy on your changes.

## Setting Up Git Hooks

To enable the pre-commit hook, run the following command:

```bash
chmod +x ./.githooks/setup-hooks.sh
./.githooks/setup-hooks.sh
```

This script will configure the hooks directory and make the pre-commit hook executable.
