"""
This module represents the Marketplace.

Computer Systems Architecture Course
Assignment 1
March 2021
"""

from threading import Lock, currentThread

class Marketplace:
    """
    Class that represents the Marketplace. It's the central part of the implementation.
    The producers and consumers use its methods concurrently.
    """
    def __init__(self, queue_size_per_producer):
        """
        Constructor

        :type queue_size_per_producer: Int
        :param queue_size_per_producer: the maximum size of a queue associated with each producer
        """
        self.queue_size_per_producer = queue_size_per_producer #size for every producer in buffer
        self.last_producer = -1 #keep track of producers ids
        self.last_cart = -1 #keep track of carts ids
        self.carts = {} #dictionary of carts, each containing items and their quantity
        self.storage = {} #dictionary of slots available for every buffer
        self.products = {} #dictionary of (product, producer_id), quantity
        self.lock_producer = Lock() #lock for producer actions
        self.lock_consumer = Lock() #lock for consumer actions

    def register_producer(self):
        """
        Returns an id for the producer that calls this.
        """
        with self.lock_producer: #access storage only with a thread
            self.last_producer += 1 #add another producer
            self.storage[self.last_producer] = self.queue_size_per_producer #initialize its slots
        return self.last_producer #return its id

    def publish(self, producer_id, product):#
        """
        Adds the product provided by the producer to the marketplace

        :type producer_id: String
        :param producer_id: producer id

        :type product: Product
        :param product: the Product that will be published in the Marketplace

        :returns True or False. If the caller receives False, it should wait and then try again.
        """
        space_available = False #return value
        with self.lock_producer: #access products and storage only with a thread
            if self.storage[producer_id] > 0: #if the producer has space
                self.storage[producer_id] -= 1 #occupy it
                if (product, producer_id) in list(self.products.keys()): #add product to dictionary
                    self.products[(product, producer_id)] += 1
                else:
                    self.products[(product, producer_id)] = 1
                space_available = True
        return space_available

    def new_cart(self):
        """
        Creates a new cart for the consumer

        :returns an int representing the cart_id
        """
        with self.lock_consumer: #access carts only with a thread
            self.last_cart += 1 #add another cart
            self.carts[self.last_cart] = {} #assign it a dictionary
        return self.last_cart

    def add_to_cart(self, cart_id, product):#
        """
        Adds a product to the given cart. The method returns

        :type cart_id: Int
        :param cart_id: id cart

        :type product: Product
        :param product: the product to add to cart

        :returns True or False. If the caller receives False, it should wait and then try again
        """
        product_exists = False #return value
        with self.lock_consumer: #access carts, products and storage only with a thread
            for (prod, producer_id) in list(self.products.keys()): #if the product exists
                if prod == product and self.products[(prod, producer_id)] > 0:
                    self.products[(prod, producer_id)] -= 1 #remove it from products
                    self.storage[producer_id] += 1 #release the space used
                    if prod in self.carts[cart_id]: #add it to cart
                        self.carts[cart_id][prod] += 1
                    else:
                        self.carts[cart_id][prod] = 1
                    product_exists = True
                    break
        return product_exists

    def remove_from_cart(self, cart_id, product):#
        """
        Removes a product from cart.

        :type cart_id: Int
        :param cart_id: id cart

        :type product: Product
        :param product: the product to remove from cart
        """
        product_exists = True
        with self.lock_consumer: #verify cart only with a thread
            if product not in self.carts[cart_id]: #if the product is not in cart
                product_exists = False

        if product_exists is False:
            return

        with self.lock_consumer: #modify carts only with a thread
            if self.carts[cart_id][product] == 1: #remove the product
                del self.carts[cart_id][product]
            else:
                self.carts[cart_id][product] -= 1
            for (prod, producer_id) in list(self.products.keys()):
                if prod == product: #add it back in products dictionary
                    self.storage[producer_id] -= 1
                    self.products[(prod, producer_id)] += 1
                    break

    def place_order(self, cart_id):#
        """
        Return a list with all the products in the cart.

        :type cart_id: Int
        :param cart_id: id cart
        """
        with self.lock_consumer: #access carts only with a cart and preserve order
            for product, quantity in self.carts[cart_id].items(): #print the contents of the cart
                for _ in range(quantity):
                    print(currentThread().getName() + " bought ", end='')
                    print(product)
