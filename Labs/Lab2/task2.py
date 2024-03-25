import threading, random
from threading import Thread

def printThread(i, num):
    print("Hello, I'm Thread-" + str(i) + " and I received the number " + str(num))

def main():
    NO_THREADS = int(input())
    t_list = []

    for i in range(NO_THREADS):
        num = random.randint(0, 100)
        t = Thread(target = printThread, args = (i, num))
        t.start()
        t_list.append(t)
    
    for i in range(NO_THREADS):
        t_list[i].join()


if __name__ == '__main__':
    main()