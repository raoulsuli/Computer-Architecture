from os import listdir

RECIPES_FOLDER = "recipes"

def load_recipes():

	recipes = {}
	for file in listdir(RECIPES_FOLDER):
		f = open(RECIPES_FOLDER + "/" + file)
		
		items = f.read().split("\n")
		coffee = items[0]
		recipes[coffee] = {}
		items = items[1:]
		for i in items:
			if (i == ''): continue
			item = i.split("=")
			recipes[coffee][item[0]] = int(item[1])
	return recipes
