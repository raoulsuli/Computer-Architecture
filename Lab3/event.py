from threading import enumerate, Event, Thread, Condition

class Master(Thread):
    def __init__(self, max_work, work_result):
        Thread.__init__(self, name = "Master")
        self.max_work = max_work
        self.work_result = work_result
    
    def set_worker(self, worker):
        self.worker = worker
    
    def run(self):
        for i in range(self.max_work):
            with self.work_result:
                self.work = i
                self.work_result.notify()
                self.work_result.wait()
                if self.get_work() + 1 != self.worker.get_result():
                    print ("oops")
                print ("%d -> %d" % (self.work, self.worker.get_result()))
    
    def get_work(self):
        return self.work

class Worker(Thread):
    def __init__(self, terminate, work_result):
        Thread.__init__(self, name = "Worker")
        self.terminate = terminate
        self.work_result = work_result

    def set_master(self, master):
        self.master = master
    
    def run(self):
        while(True):
            with self.work_result:
                self.work_result.wait()
                if(terminate.is_set()): break
                self.result = self.master.get_work() + 1
                self.work_result.notify()
    
    def get_result(self):
        return self.result

if __name__ ==  "__main__":
    # create shared objects
    terminate = Event()
    work_result = Condition()
    
    # start worker and master
    w = Worker(terminate, work_result)
    m = Master(10, work_result)
    w.set_master(m)
    m.set_worker(w)
    w.start()
    m.start()

    # wait for master
    m.join()

    # wait for worker
    with work_result:
        terminate.set()
        work_result.notify()
    
    w.join()

    # print running threads for verification
    print(enumerate())

