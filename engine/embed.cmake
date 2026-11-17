file(READ ${IN} hex HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," hex "${hex}")
file(WRITE ${OUT} "#pragma once\nnamespace fe {\nstatic const char kKernelSource[] = {${hex}0x00};\n}\n")
