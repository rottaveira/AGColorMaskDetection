import ast
import math
import random
import numpy as np
import cv2
from scipy.ndimage import convolve

IMAGE_PATH = f"image.jpg"
MASK_PATH = f"train_mask.jpg"
img=0
population_size = 100
POPULATION=[]
TX_CROSSOVER = math.ceil(population_size *.6)
TX_PARENTS = math.ceil(population_size*.4)
TX_MUTATION = math.ceil(population_size * 0.06)
MAX_GENERATIONS = 100
num_best_individuals = 5
population_filename = 'population.txt'

'''
The Dice coefficient ranges from 0 to 1, with a value of 1 indicating a perfect overlap 
between the two images. Higher values indicate better similarity or agreement between the images.
'''
def calculate_dice_coefficient(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    dice_coefficient = (2.0 * intersection.sum()) / (mask1.sum() + mask2.sum())
    return dice_coefficient

def evaluate(individuo, final = False):
    global img
    mask = cv2.imread(MASK_PATH)
    image = cv2.imread(IMAGE_PATH)

    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper boundaries for the color in HSV
    lower = individuo[0] # Lower boundary
    upper = individuo[1] # Upper boundary 

    # Create the color mask
    collor_mask = cv2.inRange(hsv_image, lower, upper)
   
    # Create a white image of the same size as the original image
    white_image = np.full_like(image, (255, 255, 255), dtype=np.uint8)

    # Copy the masked colored points from the original image to the white image
    result = cv2.bitwise_or(white_image, white_image, mask=collor_mask)
    
    if final:
        result = cv2.bitwise_and(image, image, mask=collor_mask)
        cv2.imwrite(f'Final Image{img}.jpg', result)
        return 1
    else:
        # save the masked image
        cv2.imwrite(f'Segmented Image{img}.jpg', result)        
        img+=1
        dice_coefficient = calculate_dice_coefficient(result,mask)
        return dice_coefficient
 
def roulette(samples, qtdparentsSamples):
    roulette_sum = sum(samples.values())
    roullete =[]
    start = 0
    
    for n in samples:
        new_start = abs(samples[n]) + start
        roullete.append([start,new_start])
        start = new_start
    
    parents = [] 
    while len(parents) < qtdparentsSamples: #stop when reach in x pairs (dont remove parents)
        parent = random.uniform(0,roulette_sum)        
        for n in roullete: 
            if  parent >= n[0] and parent <= n[1]:
                item = POPULATION[roullete.index(n)]
                parents.append(item)
                break
         
    return parents
 
def generate_random_color_mask(population_size):
    populacao_inicial = []
    for _ in range(population_size):
        lower_boundary = np.random.randint(0, 255, size=(3,))
        upper_boundary = np.random.randint(0, 255, size=(3,))
        populacao_inicial.append((lower_boundary, upper_boundary))
    return populacao_inicial


def getMutationValue(len):
    value = random.randint(0,255)
    position = random.randint(0, len)
    return value,position

def mutation(childrens): 

    indexes = random.sample(range(0, len(childrens)), TX_MUTATION)

    for index in indexes:
        element = childrens[index]
        #lower
        value,position = getMutationValue(len(element[0])-1)
        element[0][position] = value
        #upper
        value,position = getMutationValue(len(element[0])-1)
        element[1][position] = value
        
    return childrens

def crossover(parents):
    childrens = []
    while(len(childrens) < TX_CROSSOVER):
      
        players = random.sample(parents, 2) 
        parent1 = players[0] 
        parent2 = players[1] 
          
        crossover_point = np.random.randint(1, len(parent1))  # Randomly choose a crossover point
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]), axis=0)
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]), axis=0)

        childrens.append(child1)
        childrens.append(child2)

    return mutation(childrens) #0.06

def trainModel(): 
    global POPULATION
    global img
    # Generate initial population
    POPULATION = generate_random_color_mask(population_size)
 
    for n in range(MAX_GENERATIONS):
        img = 0
        print(n)
        classification ={}   
        for index, individual in enumerate(POPULATION):
            evaluate_result = evaluate(individual) 
            classification[index] = evaluate_result
         
        ordered = dict(sorted(classification.items(), key=lambda x:x[1], reverse=True))
        bestSamples = [POPULATION[sample] for sample in  list({k: ordered[k] for k in list(ordered)[:TX_PARENTS]}.keys())]
 
        parents = roulette(ordered, population_size)

        POPULATION.clear()
        POPULATION.extend(bestSamples) #40%
        POPULATION.extend(crossover(parents)) #60%

   
    # Save the best individuals
    best_individuals = POPULATION[:num_best_individuals] 

    # Convert the population to a list of tuples
    population_list = [(lower, upper) for lower, upper in best_individuals]

    # Save the population list to the text file
    with open(population_filename, 'w') as f:
        for item in population_list:
            f.write(f"{item}\n")

    # Print a message indicating the successful save
    print(f"Population saved to {population_filename}")


def testModel():
    global IMAGE_PATH
    IMAGE_PATH = f"final.jpg"
    # Load the population from the text file
    loaded_population = []
    with open(population_filename, 'r') as f:
        for line in f:
            item = eval(line.strip().replace('array', ''))
            loaded_population.append(np.array(item))  
    
    for gene in loaded_population:
        evaluate(gene, True)
    

if __name__ == '__main__':     
    trainModel();    
    #testModel();


