# CS595-Project-FedRL
Application Federated Reinforcement Learning

## Installation

install all dependencies from the main directory:

```bash
cd ~/Desktop/CS595-Project-FedRL
pip install -r requirements.txt
```

**note:** all requirements are consolidated in the main `requirements.txt` file. subdirectories no longer have separate requirements files.

---

## Project Structure

- **moonlanderv1/** - initial implementation
- **moonlanderv2/** - improved version
- **moonlanderv3/** - persistent environments + disk logging
- **moonlanderv4/** - gradual weight adjustment (latest)

see `moonlanderv4/INSTRUCTIONS.md` for detailed usage instructions.

---

## TODO:

✅ implement gradual weight adjustment mechanism (done in v4)
✅ implement differential privacy step (done in v3)