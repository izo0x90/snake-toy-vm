<p align="center">
  <img src="logo2.png" width="820">
</p>

# Snake toy VM :snake:

A generalized toy virtual machine, assembler, debugger/ UI written in python. Intended as just a fun time waster over a few weekends for myself, and also somehow we dragged in [@zayfod](https://github.com/zayfod) into this too, but maybe there is some educational value to others?

- Keeping things simple, illustrating basic ideas behind classical computing, emulation and VMs
- Modular way to define architecture/ VMs
  - Allows for definition of custom assembly sytaxes
  - Swapping/ extending instruction sets
- VMs currently in project
  - Toy VM (In progress)
  - Brainfuck (Completed) ✔️
  - F4 MISc (Completed) ✔️
  - 8 bit PIC (In progress)
 
*It's all virtual machines in virtual machins in virtual machines ... and soon Transformer models infrencing the output of virtual machines ... - Probaly someone*

<p align="center">
  <img src="screen_shot.png" width="500">
</p>

## Why?
Writing a VM in a language that runs in a VM and has no business having a VM written in :feelsgood:! Possibly educational for someone :suspect:? If you grew up learning to code in the 90's you get it.

## Install

### Pre
Clone repo

Install python 3.13 

Install `uv` (Optional):
```
pip install uv
```

Install deps:
Setup virtual env., and install deps. (Uses `uv`)
```
make dev_build_env
```
or
```
uv pip install -r requirements.txt
```
or
```
pip install -r requirements.txt
```

## Run
Load with only default "Toy VM" virtual machine:
```
python main.py
```
Load with addition virtual machines define in python modules:
```
python main.py 8_bit_pic f4_misc brainfuck
```
or you can pass-in the same arch. module multiple times to get more than one instance of the same VM type
```
python main.py toy_vm toy_vm
```

## Contribute
All check/ auto-fixes:
```
make check-all
```

Lint:
```
make lint
```
or 
```
ruff check
```

Format:
```
make format
```
or
```
ruff format
```

Type checks:
```
make type-check
```
or
```
pyright .
```
