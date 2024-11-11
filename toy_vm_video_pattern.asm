# Fill screen with pattern
      LOADA 0x3E80 # Offset to end of video memory
      PUSHA

LABEL .ROW_LOOP:
      # Decrement row offset
      # We can get here more efficiently if we just
      # push the result of the column iteration but
      # this is easier to read, maybe ?!
      POPA
      LOADB -320
      ADD
      PUSHA
      # Increment col offset to end of row
      LOADB 320
      ADD
      PUSHA

LABEL .COL_LOOP
      # Write second 2 bytes of pixel
      POPA
      LOADB -2
      MOVAC
      ADD
      MOVAIX
      MSTOREC 0x1000 # Video memory

      # Write first 2 bytes of pixel
      LOADC 0xFFFF
      ADD
      MOVAIX
      MSTOREC 0x1000 # Video memory

      # Remove row offset see if at COL=0
      POPB
      PUSHB
      PUSHA
      NEGB
      ADD
      JZ .EXIT_COL_LOOP
      JMP .COL_LOOP
LABEL .EXIT_COL_LOOP:
      # Check if ROW = 0
      POPA
      PUSHA
      LOADB 0
      ADD
      JZ .EXIT_CHECKERED_VIDEO_LOOP
      JMP .ROW_LOOP
LABEL .EXIT_CHECKERED_VIDEO_LOOP:
      HALT

