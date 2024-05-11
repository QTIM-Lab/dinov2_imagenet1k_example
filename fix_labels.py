# Open the input file in read mode
with open('labels_original.txt', 'r') as infile:
    # Open a new file in write mode to store the modified content
    with open('labels.txt', 'w') as outfile:
        # Iterate through each line in the input file
        for line in infile:
            # Split the line into components using whitespace as separator
            parts = line.split()
            # Write the first and third components to the output file
            outfile.write(parts[0] + ',' + parts[2] + '\n')
