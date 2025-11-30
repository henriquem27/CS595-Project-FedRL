# CS595-Project-FedRL
Application Federated Reinforcement Learning

## Installation

install docker on your system.

**ubuntu/debian:**
```bash
sudo apt update
sudo apt install -y docker.io
sudo usermod -aG docker $USER
newgrp docker
```

**centos/rhel:**
```bash
sudo yum install -y docker
sudo systemctl start docker
sudo usermod -aG docker $USER
```

---

## Quick Start

```bash
cd moonlanderv4
chmod +x run.sh
./run.sh
```

see `moonlanderv4/INSTRUCTIONS.md` for details.

---

## Project Structure

- **moonlanderv1/** - initial implementation
- **moonlanderv2/** - improved version
- **moonlanderv3/** - persistent environments + disk logging
- **moonlanderv4/** - gradual weight adjustment + docker (latest)

---

## TODO:

✅ implement gradual weight adjustment mechanism (done in v4)
✅ implement differential privacy step (done in v3)