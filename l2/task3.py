from random import randint
from threading import Lock, Thread, Semaphore

NO_PROD = 10
NO_CONS = 40
COFFEES = ['Espresso', 'Americano', 'Capuccino']
SIZES = ['small', 'medium']

class Coffee:
    def __init__(self, name, size):
        self.name = name
        self.size = size

    def get_name(self):
        return self.name

    def get_size(self):
        return self.size

    def get_message(self):
        pass


class Espresso(Coffee):
    def __init__(self, size):
        Coffee.__init__(self, "Espresso", size)

    def get_message(self):
        print("produced a nice " + self.size + " espresso")

class Americano(Coffee):
    def __init__(self, size):
        Coffee.__init__(self, "Americano", size)

    def get_message(self):
        print("produced an strong " + self.size + " americano")

class Capuccino(Coffee):
    def __init__(self, size):
        Coffee.__init__(self, "Capuccino", size)

    def get_message(self):
        print("produced an italian " + self.size + " capuccino")

# buffer
class Distributor:
    def __init__(self, size):
        self.items = []
        self.size = size
        self.lockC = Lock()
        self.lockP = Lock()
        self.semGol = Semaphore(size)
        self.semPlin = Semaphore(0)
    
    def put_coffee(self, prod_id, coffee):
        self.semGol.acquire()
        self.lockP.acquire()
        self.items.append(coffee)
        self.lockP.release()
        self.semPlin.release()
        print("Factory " + str(prod_id), end=' ')
        coffee.get_message()

    def take_coffee(self, cons_id):
        self.semPlin.acquire()
        self.lockC.acquire()
        coffee = self.items.pop(0)
        self.lockC.release()
        self.semGol.release()
        print("Consumer " + str(cons_id) + " consumed " + coffee.get_name())

# consumator
class User(Thread):
    def __init__(self, user_id, buffer):
        Thread.__init__(self)
        self.user_id = user_id
        self.buffer = buffer

    def run(self):
        while True:
            self.buffer.take_coffee(self.user_id)

# producator
class CoffeeFactory(Thread):
    def __init__(self, prod_id, buffer):
        Thread.__init__(self)
        self.prod_id = prod_id
        self.buffer = buffer

    def run(self):
        while True:
            coffeeType = COFFEES[randint(0, 2)]
            coffeeSize = SIZES[randint(0, 1)]
            if coffeeType == 'Espresso':
                coffee = Espresso(coffeeSize)
            elif coffeeType == 'Capuccino':
                coffee = Capuccino(coffeeSize)
            else:
                coffee = Americano(coffeeSize)
            self.buffer.put_coffee(self.prod_id, coffee)

def main():
    buffer = Distributor(NO_PROD)
    threads = []

    for i in range(NO_PROD):
        threads.append(CoffeeFactory(i, buffer))

    for i in range(NO_CONS):
        threads.append(User(i, buffer))

    for t in threads:
        t.start()

    for t in threads:
        t.join()


if __name__ == '__main__':
    main()
