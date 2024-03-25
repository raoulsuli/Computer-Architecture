from load_recipes import load_recipes

# Commands
EXIT = "exit"
LIST_COFFEES = "list"
MAKE_COFFEE = "make"
HELP = "help"
REFILL = "refill"
RESOURCE_STATUS = "status"
commands = [EXIT, LIST_COFFEES, MAKE_COFFEE, REFILL, RESOURCE_STATUS, HELP]

# Coffee
ESPRESSO = "espresso"
AMERICANO = "americano"
CAPPUCCINO = "cappuccino"
coffee_list = [ESPRESSO, AMERICANO, CAPPUCCINO]

# Resources
WATER = "water"
COFFEE = "coffee"
MILK = "milk"
ALL = "all"

# Coffee maker's resources
RESOURCES = {WATER: 100, COFFEE: 100, MILK: 100}

def verify_resources(resources_needed):
    
    for key, val in resources_needed.items():
        if RESOURCES[key] < val:
            return False
    return True

def make_coffee(resources_needed):

    for key, val in resources_needed.items():
        RESOURCES[key] -= val

if __name__ == "__main__":
    print("I'm a smart coffee maker")
    coffee_resources = load_recipes()

    while True:
        print("\nEnter command:")
        command = input()

        if command == HELP:
            print(commands)
        
        elif command == EXIT:
            print("Quitting...")
            break

        elif command == LIST_COFFEES:
            print(coffee_list)
        
        elif command == RESOURCE_STATUS:
            print(RESOURCES)

        elif command == MAKE_COFFEE:
            print("Which coffee?")
            coffee = input()
            
            if verify_resources(coffee_resources[coffee]):
                print("Making " + coffee)
                make_coffee(coffee_resources[coffee])
            else:
                print("Not enough resources. Refill please!")           
        
        elif command == REFILL:
            print("Which resource do you want to refill? Type 'all' for everything")

            resource = input()

            if resource == ALL:
                for key in list(RESOURCES.keys()):
                    RESOURCES[key] = 100
            elif resource in [WATER, MILK, COFFEE]:
                RESOURCES[resource] = 100

            print("Done!")

