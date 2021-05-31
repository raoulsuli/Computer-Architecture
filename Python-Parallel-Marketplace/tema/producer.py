"""
This module represents the Producer.

Computer Systems Architecture Course
Assignment 1
March 2021
"""

from threading import Thread
import time

class Producer(Thread):
    """
    Class that represents a producer.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Constructor.

        @type products: List()
        @param products: a list of products that the producer will produce

        @type marketplace: Marketplace
        @param marketplace: a reference to the marketplace

        @type republish_wait_time: Time
        @param republish_wait_time: the number of seconds that a producer must
        wait until the marketplace becomes available

        @type kwargs:
        @param kwargs: other arguments that are passed to the Thread's __init__()
        """
        Thread.__init__(self, **kwargs) #give the name of the thread
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = self.marketplace.register_producer() #assign a producer id

    def run(self):
        while True: #produce while all the consumers get their objects
            for product, quantity, sleep_time in self.products:
                i = 0
                while i < quantity: #produce all the products needed in a batch
                    production_pass = self.marketplace.publish(self.producer_id, product)
                    if production_pass is False: #if there is no space available for this producer
                        time.sleep(self.republish_wait_time) #wait
                    else:
                        time.sleep(sleep_time) #wait for production
                        i += 1 #go to the next one
