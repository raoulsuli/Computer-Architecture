from concurrent.futures import ThreadPoolExecutor
import random

def substr(lst, seq, index):
    if lst[index].find(seq) != -1:
        return "DNA sequence found in sample " + str(index)
 
if __name__ == "__main__":
    random.seed(28)

    DNA = []
    for i in range(100):
        curr = ''
        for j in range(10000):
            curr += random.choice('ATGC')
        DNA.append(curr)
    
    seq = "ATTGGCCACA"

    with ThreadPoolExecutor(max_workers = 30) as executor:
        results = []
        for i in range(100):
            results.append(executor.submit(substr, DNA, seq, i))

        for res in results:
            if res.result() != None:
                print(res.result())
        