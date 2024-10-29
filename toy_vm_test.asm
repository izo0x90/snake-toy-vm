# 16-bit word tests

LABEL .START: NOP

# JMP .END

# JMP 46 

# JMP .ONE_ADD

# No overflow or carry
LOADA 0xFFFD
LOADB 0x0002
ADD

# Signed overflow and carry
LOADA 0x8000
LOADB 0xffff
ADD

# Ex. Signed overflow 0x7fff (32767) + 0x0001 (1) and signed flag
LOADA 0x7fff
LOADB 0x0001
ADD

LABEL .ONE_ADD:
# Ex. Carry but no Signed overflow 00x8001 (-32767) + 0x7fff (32767) and zero flag
LOADA 0x8001
LOADB 0x7fff
ADD

LABEL .END:
HALT
