[MSMPI and MPI SDK](https://www.microsoft.com/en-us/download/details.aspx?id=57467) should be installed in directory of the project to be able to use MPI library.

Make sure to `${MSMPI_INC}` and `${MSMPI_LIB64}` be added into 'includePath' setting of the VSCode.

Script to compile with GCC
```bash
g++ filename.cpp -I $env:MSMPI_INC\ -L $env:MSMPI_LIB64\ -lmsmpi -o filename
```

Also can be added to tasks.json (for VSCode it's `Terminal > Configure default build task`) for automatic compilation. `args` should contain following arguments:
```json
"-fdiagnostics-color=always",
"-g",
"${file}",
"-I",
"${MSMPI_INC}",
"-L",
"${MSMPI_LIB64}",
"-lmsmpi",
"-o",
"${fileDirname}\\${fileBasenameNoExtension}.exe"
```

To run .exe file type in the terminal `mpiexec -n 4 filename` for 4 processes