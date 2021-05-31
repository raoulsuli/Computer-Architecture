# Sulimovici Raoul-Renatto 331CB
# Computer Systems Architecture Course
# Assignment 1 - Multiple Producers Multiple Consumers
# March 2021

Git: https://github.com/raoulsuli/Tema-1-ASC

Resources: Labs -> https://ocw.cs.pub.ro/courses/asc

Producer module contains run method which creates each type of product, based on the input list, in a while loop.
If the products buffer is full, the producer thread will wait and try again after the sleep time.

Consumer module contains run method which iterates through the list of items to add or delete. For delete operation, the program
removes all the item requested. For add operation, the consumer checks if the product exists in the buffer. If it does, it adds it
in the cart. Otherwise, it will wait.

Marketplace is the main module and it contains the following objects:
    -> a dictionary of carts, each containing a dictionary of products
    -> storage dictionary, to see if a specific producer has slots available in buffer
    -> a products dictionary, each dictionary having a tuple (product, producer_id) as key and a quantity for the products as value
    -> two locks, one for producer and one for consumer

The methods in marketplace are:
    -> register_producer, which increments the number of producers and assign it an id and space in buffer
    -> publish, which adds an item to the product dictionary and decreases slots available for a producer
    -> new_cart, which increments the number of carts and assigns a new cart to a consumer
    -> add_to_cart, which checks if buffer has the item wanted, makes it unavailable for other consumers, releases its slot in buffer
            and adds it to the cart
    -> remove_from_cart, which checks if the cart contains that item, removes it from the cart, adds it to the products buffer and
            reserves a slot for it
    -> place_order, which will print all the items contained in a cart