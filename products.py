

def parse_products(products):
	units = ["g", "kg", "ml"]

	def parse(name):
		tokens = name.strip().split()

		# find first token that contains a number
		first_num = len(tokens)
		for i in range(len(tokens)):
			if has_num(tokens[i]):
				first_num = i
				break

		# name is everything before that
		name = " ".join(tokens[0:first_num])
		
		


