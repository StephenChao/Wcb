from __future__ import print_function
import os
import re

years = ["2016APV", "2016", "2017", "2018","Full-Run2"]

# directories
directories = [("../../cards/%s/nTFa_3_nTFb_6/outs" % ((year))) for year in years]

# match the texts
asm_pattern = re.compile(r"Expected 50.0%: r < ([\d.]+)")
sig_pattern = re.compile(r"Significance: ([\d.]+)")
p_pattern   = re.compile(r"p-value = ([\d.]+)")

# loop through dirs
with open('results.txt', 'w') as output_file:
    for dir in directories:
        for subdir, dirs, files in os.walk(dir):
            for file in files:
                if file == "AsymptoticLimits.txt":
                    # open AsymptoticLimits files
                    with open(os.path.join(subdir, file), 'r') as f:
                        content = f.read()
                        asm_match = asm_pattern.search(content)
                        if asm_match:
                            output_line = ("Found in %s: r < %s\n" % ((os.path.join(subdir, file)), (asm_match.group(1))))
                            print(output_line, end='')
                            output_file.write(output_line)
                            
                elif file == "Significance.txt":
                    # open Significance files
                    with open(os.path.join(subdir, file), 'r') as f:
                        content = f.read()
                        sig_match = sig_pattern.search(content)
                        if sig_match:
                            output_line = ("Found in %s: Significance: %s\n" % ((os.path.join(subdir, file)), (sig_match.group(1))))
                            print(output_line, end='')
                            output_file.write(output_line)
                        p_match = p_pattern.search(content)
                        if p_match:
                            output_line = ("Found in %s: p value: %s\n" % ((os.path.join(subdir, file)), (p_match.group(1))))
                            print(output_line, end='')
                            output_file.write(output_line)
