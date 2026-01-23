import re
import numpy as np

# Parse old algorithm data
old_data = {
    'time': [],
    'stmts_per_sec': [],
    'cache_miss_rate': [],
    'branch_miss_rate': [],
    'instructions': [],
    'cycles': []
}

old_raw = """
=== Iteration 1 ===
  Time: 1.537787422s
  Statements per second: 65031.10
     <not counted>      cpu_atom/cache-misses/                                                  (0.00%)
       117,010,374      cpu_core/cache-misses/           #   73.35% of all cache refs         
     <not counted>      cpu_atom/branch-misses/                                                 (0.00%)
        47,053,965      cpu_core/branch-misses/          #    1.48% of all branches           
     <not counted>      cpu_atom/instructions/                                                  (0.00%)
    17,588,620,796      cpu_core/instructions/           #    1.57  insn per cycle            
     <not counted>      cpu_atom/cycles/                                                        (0.00%)
    11,222,482,529      cpu_core/cycles/                                                      
=== Iteration 2 ===
  Time: 1.551910337s
  Statements per second: 64439.29
       158,250,142      cpu_atom/cache-misses/           #   60.66% of all cache refs           (0.34%)
       117,038,221      cpu_core/cache-misses/           #   74.05% of all cache refs           (99.66%)
        43,719,006      cpu_atom/branch-misses/          #    1.73% of all branches             (0.34%)
        46,943,172      cpu_core/branch-misses/          #    1.47% of all branches             (99.66%)
    12,476,784,631      cpu_atom/instructions/           #    1.41  insn per cycle              (0.34%)
    17,594,920,273      cpu_core/instructions/           #    1.55  insn per cycle              (99.66%)
     8,856,723,752      cpu_atom/cycles/                                                        (0.34%)
    11,346,961,089      cpu_core/cycles/                                                        (99.66%)
=== Iteration 3 ===
  Time: 1.609563696s
  Statements per second: 62131.12
        18,211,372      cpu_atom/cache-misses/           #   21.53% of all cache refs           (0.00%)
       119,373,111      cpu_core/cache-misses/           #   76.79% of all cache refs           (100.00%)
        62,978,647      cpu_atom/branch-misses/          #    4.49% of all branches             (0.00%)
        46,958,731      cpu_core/branch-misses/          #    1.47% of all branches             (100.00%)
     7,187,515,652      cpu_atom/instructions/           #    0.84  insn per cycle              (0.00%)
    17,601,429,783      cpu_core/instructions/           #    1.51  insn per cycle              (100.00%)
     8,576,992,631      cpu_atom/cycles/                                                        (0.00%)
    11,687,217,801      cpu_core/cycles/                                                        (100.00%)
=== Iteration 4 ===
  Time: 1.552088778s
  Statements per second: 64431.88
       127,367,027      cpu_atom/cache-misses/           #   59.46% of all cache refs           (0.92%)
       117,781,431      cpu_core/cache-misses/           #   73.75% of all cache refs           (99.08%)
        25,393,307      cpu_atom/branch-misses/          #    1.47% of all branches             (0.92%)
        47,325,687      cpu_core/branch-misses/          #    1.48% of all branches             (99.08%)
     9,310,963,279      cpu_atom/instructions/           #    1.05  insn per cycle              (0.92%)
    17,654,322,598      cpu_core/instructions/           #    1.56  insn per cycle              (99.08%)
     8,873,891,815      cpu_atom/cycles/                                                        (0.92%)
    11,345,320,067      cpu_core/cycles/                                                        (99.08%)
=== Iteration 5 ===
  Time: 1.543745558s
  Statements per second: 64780.11
        14,736,910      cpu_atom/cache-misses/           #    7.87% of all cache refs           (0.00%)
       117,454,623      cpu_core/cache-misses/           #   73.70% of all cache refs           (100.00%)
        60,573,921      cpu_atom/branch-misses/          #    5.01% of all branches             (0.00%)
        46,939,719      cpu_core/branch-misses/          #    1.47% of all branches             (100.00%)
     6,262,644,756      cpu_atom/instructions/           #    0.74  insn per cycle              (0.00%)
    17,583,423,449      cpu_core/instructions/           #    1.56  insn per cycle              (100.00%)
     8,413,450,627      cpu_atom/cycles/                                                        (0.00%)
    11,291,706,359      cpu_core/cycles/                                                        (100.00%)
=== Iteration 6 ===
  Time: 1.545473848s
  Statements per second: 64707.66
       139,806,963      cpu_atom/cache-misses/           #   57.49% of all cache refs           (0.37%)
       118,046,968      cpu_core/cache-misses/           #   73.73% of all cache refs           (99.63%)
        51,414,532      cpu_atom/branch-misses/          #    1.92% of all branches             (0.37%)
        47,031,409      cpu_core/branch-misses/          #    1.47% of all branches             (99.63%)
    13,567,154,500      cpu_atom/instructions/           #    1.53  insn per cycle              (0.37%)
    17,625,330,887      cpu_core/instructions/           #    1.55  insn per cycle              (99.63%)
     8,871,996,882      cpu_atom/cycles/                                                        (0.37%)
    11,334,784,968      cpu_core/cycles/                                                        (99.63%)
=== Iteration 7 ===
  Time: 1.560120331s
  Statements per second: 64100.18
       210,521,041      cpu_atom/cache-misses/           #   91.84% of all cache refs           (0.18%)
       117,775,710      cpu_core/cache-misses/           #   73.99% of all cache refs           (99.82%)
         6,774,400      cpu_atom/branch-misses/          #    0.34% of all branches             (0.18%)
        47,179,390      cpu_core/branch-misses/          #    1.48% of all branches             (99.82%)
     8,277,060,617      cpu_atom/instructions/           #    0.93  insn per cycle              (0.18%)
    17,585,509,238      cpu_core/instructions/           #    1.54  insn per cycle              (99.82%)
     8,907,062,447      cpu_atom/cycles/                                                        (0.18%)
    11,406,382,835      cpu_core/cycles/                                                        (99.82%)
=== Iteration 8 ===
  Time: 1.546322312s
  Statements per second: 64672.16
     <not counted>      cpu_atom/cache-misses/                                                  (0.00%)
       118,445,804      cpu_core/cache-misses/           #   74.35% of all cache refs         
     <not counted>      cpu_atom/branch-misses/                                                 (0.00%)
        47,009,716      cpu_core/branch-misses/          #    1.47% of all branches           
     <not counted>      cpu_atom/instructions/                                                  (0.00%)
    17,608,784,595      cpu_core/instructions/           #    1.55  insn per cycle            
     <not counted>      cpu_atom/cycles/                                                        (0.00%)
    11,357,798,881      cpu_core/cycles/                                                      
=== Iteration 9 ===
  Time: 1.549475025s
  Statements per second: 64540.57
        17,835,059      cpu_atom/cache-misses/           #    8.13% of all cache refs           (0.00%)
       118,066,254      cpu_core/cache-misses/           #   74.35% of all cache refs           (100.00%)
        60,535,254      cpu_atom/branch-misses/          #    5.50% of all branches             (0.00%)
        46,954,306      cpu_core/branch-misses/          #    1.47% of all branches             (100.00%)
     6,088,812,615      cpu_atom/instructions/           #    0.73  insn per cycle              (0.00%)
    17,579,461,644      cpu_core/instructions/           #    1.55  insn per cycle              (100.00%)
     8,292,646,019      cpu_atom/cycles/                                                        (0.00%)
    11,340,336,814      cpu_core/cycles/                                                        (100.00%)
=== Iteration 10 ===
  Time: 1.557086828s
  Statements per second: 64225.06
       149,326,334      cpu_atom/cache-misses/           #   60.49% of all cache refs           (0.35%)
       119,724,658      cpu_core/cache-misses/           #   74.56% of all cache refs           (99.65%)
        47,655,091      cpu_atom/branch-misses/          #    1.81% of all branches             (0.35%)
        47,060,292      cpu_core/branch-misses/          #    1.48% of all branches             (99.65%)
    13,110,679,107      cpu_atom/instructions/           #    1.47  insn per cycle              (0.35%)
    17,606,118,067      cpu_core/instructions/           #    1.54  insn per cycle              (99.65%)
     8,925,418,671      cpu_atom/cycles/                                                        (0.35%)
    11,424,695,075      cpu_core/cycles/                                                        (99.65%)
"""

# Parse new algorithm data
new_data = {
    'time': [],
    'stmts_per_sec': [],
    'cache_miss_rate': [],
    'branch_miss_rate': [],
    'instructions': [],
    'cycles': []
}

new_raw = """
=== Iteration 1 ===
  Time: 1.477882725s
  Statements per second: 67667.07
        56,452,742      cpu_atom/cache-misses/           #   78.55% of all cache refs           (0.02%)
        95,564,222      cpu_core/cache-misses/           #   73.10% of all cache refs           (99.98%)
        22,535,503      cpu_atom/branch-misses/          #    3.69% of all branches             (0.02%)
        46,107,425      cpu_core/branch-misses/          #    1.54% of all branches             (99.98%)
     3,011,376,608      cpu_atom/instructions/           #    0.38  insn per cycle              (0.02%)
    16,710,242,001      cpu_core/instructions/           #    1.61  insn per cycle              (99.98%)
     8,008,778,920      cpu_atom/cycles/                                                        (0.02%)
    10,388,550,284      cpu_core/cycles/                                                        (99.98%)
=== Iteration 2 ===
  Time: 1.48763901s
  Statements per second: 67223.30
       100,484,944      cpu_atom/cache-misses/           #   57.40% of all cache refs           (1.10%)
        95,658,851      cpu_core/cache-misses/           #   73.30% of all cache refs           (98.90%)
        26,153,619      cpu_atom/branch-misses/          #    1.55% of all branches             (1.10%)
        46,527,352      cpu_core/branch-misses/          #    1.55% of all branches             (98.90%)
     9,201,165,218      cpu_atom/instructions/           #    1.12  insn per cycle              (1.10%)
    16,758,692,479      cpu_core/instructions/           #    1.60  insn per cycle              (98.90%)
     8,210,550,724      cpu_atom/cycles/                                                        (1.10%)
    10,495,580,296      cpu_core/cycles/                                                        (98.90%)
=== Iteration 3 ===
  Time: 1.474369444s
  Statements per second: 67828.32
        28,161,191      cpu_atom/cache-misses/           #   29.65% of all cache refs           (0.02%)
        95,045,277      cpu_core/cache-misses/           #   73.12% of all cache refs           (99.98%)
        52,505,303      cpu_atom/branch-misses/          #    3.84% of all branches             (0.02%)
        46,171,578      cpu_core/branch-misses/          #    1.55% of all branches             (99.98%)
     7,483,433,811      cpu_atom/instructions/           #    0.93  insn per cycle              (0.02%)
    16,681,259,456      cpu_core/instructions/           #    1.61  insn per cycle              (99.98%)
     8,005,090,252      cpu_atom/cycles/                                                        (0.02%)
    10,384,830,156      cpu_core/cycles/                                                        (99.98%)
=== Iteration 4 ===
  Time: 1.472902932s
  Statements per second: 67895.85
       138,083,992      cpu_atom/cache-misses/           #   59.56% of all cache refs           (0.38%)
        95,060,208      cpu_core/cache-misses/           #   72.55% of all cache refs           (99.62%)
        43,145,376      cpu_atom/branch-misses/          #    1.80% of all branches             (0.38%)
        46,276,161      cpu_core/branch-misses/          #    1.55% of all branches             (99.62%)
    11,967,279,014      cpu_atom/instructions/           #    1.47  insn per cycle              (0.38%)
    16,713,654,369      cpu_core/instructions/           #    1.61  insn per cycle              (99.62%)
     8,132,074,148      cpu_atom/cycles/                                                        (0.38%)
    10,401,171,598      cpu_core/cycles/                                                        (99.62%)
=== Iteration 5 ===
  Time: 1.475515606s
  Statements per second: 67775.63
       146,881,437      cpu_atom/cache-misses/           #   64.39% of all cache refs           (0.36%)
        95,522,488      cpu_core/cache-misses/           #   72.88% of all cache refs           (99.64%)
        38,190,147      cpu_atom/branch-misses/          #    1.66% of all branches             (0.36%)
        46,310,073      cpu_core/branch-misses/          #    1.55% of all branches             (99.64%)
    11,282,001,784      cpu_atom/instructions/           #    1.39  insn per cycle              (0.36%)
    16,698,841,111      cpu_core/instructions/           #    1.60  insn per cycle              (99.64%)
     8,139,646,081      cpu_atom/cycles/                                                        (0.36%)
    10,412,235,221      cpu_core/cycles/                                                        (99.64%)
=== Iteration 6 ===
  Time: 1.497066531s
  Statements per second: 66799.97
        86,452,484      cpu_atom/cache-misses/           #   54.36% of all cache refs           (1.41%)
        96,244,785      cpu_core/cache-misses/           #   72.82% of all cache refs           (98.59%)
        29,383,790      cpu_atom/branch-misses/          #    1.65% of all branches             (1.41%)
        46,948,768      cpu_core/branch-misses/          #    1.56% of all branches             (98.59%)
     9,782,204,061      cpu_atom/instructions/           #    1.18  insn per cycle              (1.41%)
    16,784,412,783      cpu_core/instructions/           #    1.59  insn per cycle              (98.59%)
     8,275,530,894      cpu_atom/cycles/                                                        (1.41%)
    10,567,508,041      cpu_core/cycles/                                                        (98.59%)
=== Iteration 7 ===
  Time: 1.475899579s
  Statements per second: 67758.00
       150,991,645      cpu_atom/cache-misses/           #   61.13% of all cache refs           (0.37%)
        95,146,149      cpu_core/cache-misses/           #   72.29% of all cache refs           (99.63%)
        41,405,570      cpu_atom/branch-misses/          #    1.74% of all branches             (0.37%)
        46,339,746      cpu_core/branch-misses/          #    1.55% of all branches             (99.63%)
    11,774,342,848      cpu_atom/instructions/           #    1.45  insn per cycle              (0.37%)
    16,687,727,730      cpu_core/instructions/           #    1.60  insn per cycle              (99.63%)
     8,146,201,344      cpu_atom/cycles/                                                        (0.37%)
    10,428,654,095      cpu_core/cycles/                                                        (99.63%)
=== Iteration 8 ===
  Time: 1.477317782s
  Statements per second: 67692.95
       140,470,204      cpu_atom/cache-misses/           #   80.36% of all cache refs           (0.06%)
        96,005,754      cpu_core/cache-misses/           #   73.05% of all cache refs           (99.94%)
        17,654,537      cpu_atom/branch-misses/          #    1.08% of all branches             (0.06%)
        46,194,001      cpu_core/branch-misses/          #    1.55% of all branches             (99.94%)
     7,202,532,124      cpu_atom/instructions/           #    0.89  insn per cycle              (0.06%)
    16,680,599,488      cpu_core/instructions/           #    1.60  insn per cycle              (99.94%)
     8,126,040,154      cpu_atom/cycles/                                                        (0.06%)
    10,426,800,051      cpu_core/cycles/                                                        (99.94%)
=== Iteration 9 ===
  Time: 1.476609842s
  Statements per second: 67725.41
       147,985,715      cpu_atom/cache-misses/           #   67.50% of all cache refs           (0.36%)
        95,696,882      cpu_core/cache-misses/           #   72.47% of all cache refs           (99.64%)
        39,891,976      cpu_atom/branch-misses/          #    1.69% of all branches             (0.36%)
        46,311,830      cpu_core/branch-misses/          #    1.55% of all branches             (99.64%)
    11,610,554,500      cpu_atom/instructions/           #    1.42  insn per cycle              (0.36%)
    16,693,257,219      cpu_core/instructions/           #    1.61  insn per cycle              (99.64%)
     8,153,354,496      cpu_atom/cycles/                                                        (0.36%)
    10,394,111,417      cpu_core/cycles/                                                        (99.64%)
=== Iteration 10 ===
  Time: 1.477573597s
  Statements per second: 67681.23
     <not counted>      cpu_atom/cache-misses/                                                  (0.00%)
        95,682,447      cpu_core/cache-misses/           #   73.25% of all cache refs         
     <not counted>      cpu_atom/branch-misses/                                                 (0.00%)
        46,223,679      cpu_core/branch-misses/          #    1.55% of all branches           
     <not counted>      cpu_atom/instructions/                                                  (0.00%)
    16,662,329,522      cpu_core/instructions/           #    1.60  insn per cycle            
     <not counted>      cpu_atom/cycles/                                                        (0.00%)
    10,428,485,310      cpu_core/cycles/                                                       
"""

def parse_data(raw_text, data_dict):
    lines = raw_text.strip().split('\n')
    current_time = None
    current_stmts = None
    
    for line in lines:
        # Parse time
        time_match = re.search(r'Time: (\d+\.\d+)s', line)
        if time_match:
            current_time = float(time_match.group(1))
            continue
            
        # Parse statements per second
        stmts_match = re.search(r'Statements per second: (\d+\.\d+)', line)
        if stmts_match:
            current_stmts = float(stmts_match.group(1))
            data_dict['time'].append(current_time)
            data_dict['stmts_per_sec'].append(current_stmts)
            continue
            
        # Parse cache miss rate (prefer cpu_core data)
        cache_match = re.search(r'cpu_core/cache-misses/.*?# +(\d+\.\d+)%', line)
        if cache_match:
            data_dict['cache_miss_rate'].append(float(cache_match.group(1)))
            continue
            
        # Parse branch miss rate (prefer cpu_core data)
        branch_match = re.search(r'cpu_core/branch-misses/.*?# +(\d+\.\d+)%', line)
        if branch_match:
            data_dict['branch_miss_rate'].append(float(branch_match.group(1)))
            continue
            
        # Parse instructions (prefer cpu_core data)
        inst_match = re.search(r'cpu_core/instructions/.*?(\d+[,\d]*)', line)
        if inst_match:
            # Extract the number that appears before "cpu_core/instructions/"
            num_match = re.search(r'(\d+[,\d]*)\s+cpu_core/instructions/', line)
            if num_match:
                data_dict['instructions'].append(int(num_match.group(1).replace(',', '')))
            continue
            
        # Parse cycles (prefer cpu_core data)
        cycles_match = re.search(r'cpu_core/cycles/.*?(\d+[,\d]*)', line)
        if cycles_match:
            # Extract the number that appears before "cpu_core/cycles/"
            num_match = re.search(r'(\d+[,\d]*)\s+cpu_core/cycles/', line)
            if num_match:
                data_dict['cycles'].append(int(num_match.group(1).replace(',', '')))
            continue

parse_data(old_raw, old_data)
parse_data(new_raw, new_data)

def print_stats(name, data):
    print(f"\n{name}:")
    for key, values in data.items():
        if values:  # Only print if we have data
            mean = np.mean(values)
            std = np.std(values, ddof=1)
            print(f"  {key}: {mean:.3f} ± {std:.3f}")

print_stats("OLD ALGORITHM", old_data)
print_stats("NEW ALGORITHM", new_data)

# Calculate improvements
print(f"\nIMPROVEMENTS:")
old_time_mean = np.mean(old_data['time'])
new_time_mean = np.mean(new_data['time'])
time_improvement = (old_time_mean - new_time_mean) / old_time_mean * 100
print(f"Time: {time_improvement:+.2f}% ({'faster' if time_improvement > 0 else 'slower'})")

old_stmts_mean = np.mean(old_data['stmts_per_sec'])
new_stmts_mean = np.mean(new_data['stmts_per_sec'])
stmts_improvement = (new_stmts_mean - old_stmts_mean) / old_stmts_mean * 100
print(f"Statements/sec: {stmts_improvement:+.2f}%")

old_inst_mean = np.mean(old_data['instructions'])
new_inst_mean = np.mean(new_data['instructions'])
inst_reduction = (old_inst_mean - new_inst_mean) / old_inst_mean * 100
print(f"Instructions: {inst_reduction:+.2f}% reduction")

old_branch_mean = np.mean(old_data['branch_miss_rate'])
new_branch_mean = np.mean(new_data['branch_miss_rate'])
branch_improvement = (old_branch_mean - new_branch_mean) / old_branch_mean * 100
print(f"Branch miss rate: {branch_improvement:+.2f}% improvement")