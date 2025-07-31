import os

def getAnalysisNumbers():
    """
    """
    filename = 'analysisRunNumbers'

    if not os.path.exists(filename):
        with open(filename, "w") as file:
            file.write('-1')
            print(f"File '{filename}' created with default -1.")
            return [-1]

    allRunnos = []
    try:
        with open(filename, 'r') as file:
            for line in file:

                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                try:
                    runNo = int(line.strip())
                    if runNo == -1:
                        continue
                    allRunnos.append(runNo)   
                    
                except ValueError:
                    print(f"Warning. Skipping non-integer line in '{filename}'.")
                    
    except Exception as e:
        print(f"An unexpected error occurred while reading '{filename}': {e}")
        return None

    return allRunnos