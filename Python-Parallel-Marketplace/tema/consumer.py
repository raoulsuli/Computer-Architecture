"""
This module represents the Consumer.

Computer Systems Architecture Course
Assignment 1
March 2021
"""

from threading import Thread
import time

class Consumer(Thread):
    """
    Class that represents a consumer.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Constructor.

        :type carts: List
        :param carts: a list of add and remove operations

        :type marketplace: Marketplace
        :param marketplace: a reference to the marketplace

        :type retry_wait_time: Time
        :param retry_wait_time: the number of seconds that a producer must wait
        until the Marketplace becomes available

        :type kwargs:
        :param kwargs: other arguments that are passed to the Thread's __init__()
        """
        Thread.__init__(self, **kwargs) #give the name of the thread
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        for carts in self.carts: #for every sequence of operations
            cart_id = self.marketplace.new_cart() #assign a new cart id
            for cart in carts: #for every operation
                if cart['type'] == 'add':
                    i = 0
                    while i < cart['quantity']: #add all the items requested
                        consumer_pass = self.marketplace.add_to_cart(cart_id, cart['product'])
                        if consumer_pass is False: #the object did not exist
                            time.sleep(self.retry_wait_time) #wait
                        else:
                            i += 1 #go the the next one
                else: #remove operation
                    i = 0
                    while i < cart['quantity']:
                        self.marketplace.remove_from_cart(cart_id, cart['product'])
                        i += 1
            self.marketplace.place_order(cart_id) #after processing everything, order the cart
