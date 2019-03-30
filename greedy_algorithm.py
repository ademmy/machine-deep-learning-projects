#implementing greedy optimization algorithm 
class Food(object):
    #Setting all the necessary parameters
    def __init__(self,n,v,w):
        self.name = n
        self.value = v
        self.calories = w

    def getValue(self):
        return self.value

    def getCost(self):
        return self.calories

    def density(self):
        return self.getValue()/self.getCost()

    def __str__(self):
        return self.name  + ": <" +str(self.value)\
               + ', ' + str(self.calories) + '>'


def buildMenu(names, values, calories):
    #builds the menu containing all the necessary parameter
    #returns list of foods

    menu= []
    for i in range(len(values)):
        menu.append(Food(names[i], values[i],
                         calories[i]))
    return menu

def greedy(items, maxCost, keyFunction):
    """ assumes items a list,  maxcost>= 0,
        keyFunction maps elements of items to numbers """
    itemsCopy= sorted(items, key= keyFunction, reverse= False)
    result= []
    totalValue, totalCost= 0.0,0.0

    for i in range(len(itemsCopy)):
        if (totalCost +itemsCopy[i].getCost()) <=maxCost:
            result.append(itemsCopy[i])
            totalCost += itemsCopy[i].getCost()
            totalValue += itemsCopy[i].getValue()
    return(result, totalValue)

def testGreedy(items, constraint, keyFunction):
    taken, val= greedy(items, constraint, keyFunction)
    print('Total value of items taken= ', val)
    for item in taken:
        print(" ", item)


def testGreedys(foods,maxUnits):
    print('use greedy by value to allocate', maxUnits, 'calories')
    testGreedy(foods,maxUnits, Food.getValue)

    print('use greedy by cost to allocate', maxUnits,'Calories')
    testGreedy(foods,maxUnits,
               lambda x: 1/Food.getCost(x))

    print('/n Use greedy by density to allocate',maxUnits,
          'calories')
    testGreedy(foods, maxUnits, Food.density)


names=['wine','beer','pizza','burger','meatpie',
       'donut','apple','cake']
values= [89,90,95,100,90,79,50,10]
calories= [133,450,89,99,450,300,89,90]
foods= buildMenu(names, values, calories)
testGreedys(foods, 1000)

    

                

    
        
