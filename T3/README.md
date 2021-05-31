# Sulimovici Raoul-Renatto 331CB
# Tema 3 ASC
## Stocare
Hashtable-ul va fi stocat in VRAM, folosind structura hash_pair
## Implementare
* Am creat o structura hash_pair, care va retine 2 int-uri: cheia si valoarea.
* Am adaugat in clasa GpuHashTable: un camp de size (numarul curent de elemente din hashtable), un camp de capacity si un vector de hash_pair (hashtable-ul).
* In constructor, am initializat campurile clasei si am alocat memorie pe device-uri (VRAM) cu cudaMalloc pentru hashtable, pe care am populat-o dupa cu 0.
* load_factor va returna raportul size / capacity
* getBatch() va aloca cheile pe device, va face memset pentru a pune valorile si va aloca un vector de valori managed intre GPU si RAM, initializat cu 0. Va calcula num_blocks si va apela __global__ function get_batch.
* __global__ get_batch va calcula index-ul actual, va verifica ca valoarea sa nu fie mai mare decat numKeys si sa nu fie o cheie invalida. Daca cerintele sunt indeplinite, se calculeaza hash-ul pentru cheia curenta si se intra intr-un while care va cauta cheia iterativ prin Linear Probing si va salva valoarea dorita in values. (Daca nu o gaseste pe hash-ul actual, va incrementa hash-ul pana gaseste valoarea).
* insertBatch() va aloca si va copia cheile si valorile in VRAM si va salva o variabila "no_inserts", unde se va retine numarul de chei noi (nu update-uri). Se verifica load factor-ul curent, la care adaugam cheile care trebuie adaugate si se updateaza la nevoie folosind reshape(). Se apeleaza functia __global__ insert_batch si se updateaza size-ul, dupa un synchronize.
* __global__ insert_batch va functiona foarte asemanator cu get_batch(). Diferenta fiind in while, unde folosing atomicCAS se verifica cheia curenta si rezulta trei cazuri: daca este 0, se updateaza noua cheie si valoare; daca este egala cu cheia data, se updateaza valoarea; altfel se trece mai departe folosind Linear Probing.
* reshape va aloca in VRAM un nou hashtable si va apela functia __global__ reshape_hashtable. Dupa finalizare, va updata hashtable-ul cu cel prelucrat in functia reshape_hashtable.
* __global__ reshape_hashtable va parcurge hashtable-ul ca pana acum, verificand valoarea 0 in hashtable. Daca este 0 trece mai departe, daca nu este 0, o va copia in noul hashtable.
* functia hash_function() va returna hash-ul pentru cheia data. Va inmulti cheia cu un numar prim si se va face mod cu un numar prim, totul in modul, mod capacitatea curenta. Numerele prime au fost generate random si am selectat unele mari, pentru a include numarul de valori din hashtable.
* Voi apela reshape atunci cand load_factor-ul este 85% si il voi face 75%.
## Rezultate
### Rezultatele obtinute dupa rularea script-ului bench.py pe coada hp-sl.q
```
------- Test T1 START	----------

HASH_BATCH_INSERT   count: 1000000          speed: 89M/sec          loadfactor: 70%         
HASH_BATCH_GET      count: 1000000          speed: 97M/sec          loadfactor: 51%         
----------------------------------------------
AVG_INSERT: 89 M/sec,   AVG_GET: 97 M/sec,      MIN_SPEED_REQ: 10 M/sec 


------- Test T1 END	---------- 	 [ OK RESULT: +20 pts ]



------- Test T2 START	----------

HASH_BATCH_INSERT   count: 500000           speed: 78M/sec          loadfactor: 70%         
HASH_BATCH_INSERT   count: 500000           speed: 70M/sec          loadfactor: 70%         
HASH_BATCH_GET      count: 500000           speed: 218M/sec         loadfactor: 59%         
HASH_BATCH_GET      count: 500000           speed: 194M/sec         loadfactor: 51%         
----------------------------------------------
AVG_INSERT: 74 M/sec,   AVG_GET: 206 M/sec,     MIN_SPEED_REQ: 20 M/sec 


------- Test T2 END	---------- 	 [ OK RESULT: +20 pts ]



------- Test T3 START	----------

HASH_BATCH_INSERT   count: 125000           speed: 50M/sec          loadfactor: 70%         
HASH_BATCH_INSERT   count: 125000           speed: 50M/sec          loadfactor: 70%         
HASH_BATCH_INSERT   count: 125000           speed: 44M/sec          loadfactor: 70%         
HASH_BATCH_INSERT   count: 125000           speed: 41M/sec          loadfactor: 70%         
HASH_BATCH_INSERT   count: 125000           speed: 37M/sec          loadfactor: 70%         
HASH_BATCH_INSERT   count: 125000           speed: 39M/sec          loadfactor: 84%         
HASH_BATCH_INSERT   count: 125000           speed: 33M/sec          loadfactor: 70%         
HASH_BATCH_INSERT   count: 125000           speed: 43M/sec          loadfactor: 80%         
HASH_BATCH_GET      count: 125000           speed: 99M/sec          loadfactor: 76%         
HASH_BATCH_GET      count: 125000           speed: 125M/sec         loadfactor: 72%         
HASH_BATCH_GET      count: 125000           speed: 103M/sec         loadfactor: 69%         
HASH_BATCH_GET      count: 125000           speed: 123M/sec         loadfactor: 66%         
HASH_BATCH_GET      count: 125000           speed: 102M/sec         loadfactor: 64%         
HASH_BATCH_GET      count: 125000           speed: 124M/sec         loadfactor: 61%         
HASH_BATCH_GET      count: 125000           speed: 95M/sec          loadfactor: 59%         
HASH_BATCH_GET      count: 125000           speed: 90M/sec          loadfactor: 57%         
----------------------------------------------
AVG_INSERT: 42 M/sec,   AVG_GET: 108 M/sec,     MIN_SPEED_REQ: 40 M/sec 


------- Test T3 END	---------- 	 [ OK RESULT: +15 pts ]



------- Test T4 START	----------

HASH_BATCH_INSERT   count: 2500000          speed: 99M/sec          loadfactor: 70%         
HASH_BATCH_INSERT   count: 2500000          speed: 90M/sec          loadfactor: 70%         
HASH_BATCH_INSERT   count: 2500000          speed: 78M/sec          loadfactor: 70%         
HASH_BATCH_INSERT   count: 2500000          speed: 71M/sec          loadfactor: 70%         
HASH_BATCH_GET      count: 2500000          speed: 237M/sec         loadfactor: 64%         
HASH_BATCH_GET      count: 2500000          speed: 229M/sec         loadfactor: 59%         
HASH_BATCH_GET      count: 2500000          speed: 229M/sec         loadfactor: 55%         
HASH_BATCH_GET      count: 2500000          speed: 200M/sec         loadfactor: 51%         
----------------------------------------------
AVG_INSERT: 84 M/sec,   AVG_GET: 224 M/sec,     MIN_SPEED_REQ: 50 M/sec 


------- Test T4 END	---------- 	 [ OK RESULT: +15 pts ]



------- Test T5 START	----------

HASH_BATCH_INSERT   count: 20000000         speed: 86M/sec          loadfactor: 70%         
HASH_BATCH_INSERT   count: 20000000         speed: 43M/sec          loadfactor: 70%         
HASH_BATCH_GET      count: 20000000         speed: 179M/sec         loadfactor: 59%         
HASH_BATCH_GET      count: 20000000         speed: 75M/sec          loadfactor: 51%         
----------------------------------------------
AVG_INSERT: 64 M/sec,   AVG_GET: 127 M/sec,     MIN_SPEED_REQ: 50 M/sec 


------- Test T5 END	---------- 	 [ OK RESULT: +15 pts ]

TOTAL gpu_hashtable  85/85

```
### Rulare nvprof
```
==4576== Profiling application: ./gpu_hashtable 40000000 2 50
==4576== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   36.92%  354.73ms         3  118.24ms  9.8880us  279.00ms  insert_batch(hash_pair*, int*, int*, int*, int, int)
                   36.17%  347.51ms         8  43.439ms     960ns  87.462ms  [CUDA memcpy HtoD]
                   25.32%  243.27ms         2  121.64ms  47.000ms  196.27ms  get_batch(hash_pair*, int*, int, int*, int)
                    1.59%  15.284ms         3  5.0947ms  2.5920us  14.153ms  reshape_hashtable(hash_pair*, hash_pair*, int, int)
                    0.00%  8.9930us         6  1.4980us  1.0560us  3.3600us  [CUDA memset]
      API calls:   48.77%  614.78ms         8  76.847ms  9.6040us  279.01ms  cudaDeviceSynchronize
                   27.94%  352.17ms         8  44.021ms  12.037us  88.485ms  cudaMemcpy
                   16.71%  210.60ms        12  17.550ms  11.330us  205.47ms  cudaMalloc
                    3.15%  39.764ms         5  7.9529ms  272.71us  20.808ms  cudaMallocManaged
                    2.75%  34.638ms        15  2.3092ms  15.544us  7.7995ms  cudaFree
                    0.39%  4.9186ms         6  819.76us  40.586us  4.1411ms  cudaMemset
                    0.13%  1.6229ms         8  202.86us  68.081us  875.12us  cudaLaunch
                    0.07%  922.77us         2  461.38us  417.11us  505.66us  cuDeviceTotalMem
                    0.05%  634.06us       188  3.3720us     141ns  138.66us  cuDeviceGetAttribute
                    0.03%  357.27us         1  357.27us  357.27us  357.27us  cudaGetDeviceProperties
                    0.00%  45.077us         2  22.538us  21.052us  24.025us  cuDeviceGetName
                    0.00%  34.560us        53     652ns     247ns  1.8230us  cudaGetLastError
                    0.00%  21.691us        40     542ns     184ns  3.4780us  cudaSetupArgument
                    0.00%  7.9030us         8     987ns     344ns  2.0550us  cudaConfigureCall
                    0.00%  3.6570us         4     914ns     193ns  2.6500us  cuDeviceGet
                    0.00%  2.2240us         3     741ns     152ns  1.5460us  cuDeviceGetCount

==4576== Unified Memory profiling result:
Device "Tesla K40m (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
     938  166.79KB  4.0000KB  0.9961MB  152.7813MB  24.48099ms  Device To Host
Total CPU Page faults: 469
```
### Interpretare
* Se observa ca functia get_batch este mai rapida decat insert, asa cum ar fi trebuit.
* Atunci cand load_factor-ul este prea mare si se apeleaza functia de reshape, viteza insert-ului devine mai mica, asa cum ar fi trebuit.
* Cele mai costisitoare temporal sunt operatiile de memcpy de pe host pe device, fiind vorba de foarte mult valori, si apelurile functiei __global__ insert_batch.