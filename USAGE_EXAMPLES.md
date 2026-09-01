# Crysmapping v5 command examples

Run these commands from the directory containing `Crysmapping_v5.py`.

## 1. Default genetic algorithm

```powershell
python .\Crysmapping_v5.py --file-a .\01-POSCAR-i.vasp --file-b .\01-POSCAR-f.vasp --output-dir .\best_solutions_v5 --mode ga
```

The default determinant multiplier range is `t=1..4`; `diag_max` is selected automatically.

## 2. Stop and resume the genetic algorithm

```powershell
python .\Crysmapping_v5.py --file-a .\01-POSCAR-i.vasp --file-b .\01-POSCAR-f.vasp --output-dir .\best_solutions_v5 --mode ga --generations 60 --ga-stop-after-generation 10
```

```powershell
python .\Crysmapping_v5.py --file-a .\01-POSCAR-i.vasp --file-b .\01-POSCAR-f.vasp --output-dir .\best_solutions_v5 --mode ga --generations 60 --ga-resume
```

The GA checkpoint is `best_solutions_v5\ga_progress.json`.

## 3. Stop and resume bounded enumeration

```powershell
python .\Crysmapping_v5.py --file-a .\01-POSCAR-i.vasp --file-b .\01-POSCAR-f.vasp --output-dir .\best_solutions_v5_enum --mode enumerate --enumerate-stop-after-t 1
```

```powershell
python .\Crysmapping_v5.py --file-a .\01-POSCAR-i.vasp --file-b .\01-POSCAR-f.vasp --output-dir .\best_solutions_v5_enum --mode enumerate --enumerate-resume
```

The enumeration checkpoint is `best_solutions_v5_enum\enumeration_progress.json`.

Resume commands must use the same input files, output directory, and search/scoring settings as the original run.
