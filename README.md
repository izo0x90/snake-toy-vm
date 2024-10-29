<p align="center">
  <img src="logo.png" width="200">
</p>

# Snake toy VM

A generalized toy virtual machine, assembler and debuger written in python. Intended as just a fun time waster over a few weekends for myself, but maybe there is some educational value to others?
- Keeping things simple, illustrating basic ideas behind classical computing, emulation and vms.
- Modular
  - Allows for definition of custom assembly sytaxes
  - Swapping/ extending instruction sets
 
*It's all virtual machines in virtual machins in virtual machines ... and soon Transformer models infrencing the output of virtual machines ... - Probaly someone*
<p align="center">
  <img src="screen_shot.png" width="500">
</p>
Install:
- Clone repo
- Install python 3.13 on system or virt. env.
- Instal `uv` (Optional)
```
pip install uv
```
- Install deps
```
uv pip install -f requirements.txt
```
or
```
pip install -f requirements.txt
```

Run:
- Load with only default "Toy VM" virtual machine
```
python main.py
```
- Load with addition virtual machines define in python modules
```
python main.py 8_bit_pic some_other_vm_module
```
