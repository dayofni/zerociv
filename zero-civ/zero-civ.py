# zero-civ.py

import random
import math
import os
import time
import copy
import pprint
import csv
from perlin_noise import PerlinNoise
import markov


def clamp(value, s, l):
    if value <= s: return s
    elif value >= l: return l
    return value

def generate_diagram(x, y, nodes, seed=None):
    if not seed is None:
        random.seed(seed)
    board = [[0 for i in range(x)] for i in range(y)]
    node_locs = [(random.randint(0, y-1), random.randint(0, x-1)) for i in range(nodes)]
    for j, i in enumerate(node_locs):
        board[i[0]][i[1]] = j+1
    new_board = []
    for r, row in enumerate(board):
        new_board.append([])
        for c, column in enumerate(row):
            result = (0, float("inf")) # name, result
            for n, node in enumerate(node_locs):
                if result[1] > math.hypot(node[1]-r, node[0]-c): # if result is larger than the distance between
                    result = (n, math.hypot(node[1]-r, node[0]-c))
            new_board[-1].append(result[0]+1)
    return new_board

def random_walk(voronoi, steps, drunks, fallover=False, debug=False):
    # Create drunk positions.
    drunks = [[random.randint(0, len(voronoi[0])-1), random.randint(0, len(voronoi)-1)] for i in range(drunks)]
    failed_drunks = []
    # Stores current "drunken walk map". Drunks can't go back.
    walk_map = [[None for i in range(len(voronoi[0]))] for i in range(len(voronoi))]
    for _ in range(steps):
        # Every drunk acts the same number of steps.
        for d, drunk_pos in enumerate(drunks):
            # Check to see the drunk hadn't cornered itself
            if d in failed_drunks: continue
            drunk_x, drunk_y = drunk_pos[0], drunk_pos[1]
            # Set position as voronoi number. Mainly "For Convienice!" TM.
            walk_map[drunk_y][drunk_x] = voronoi[drunk_y][drunk_x]
            movement = []
            # Four cardinal directions
            checked = [0, 1, 2, 3]
            while movement == []:
                # Direction around compass rose. 0 = N, 1 = E etc.
                if checked == []:
                    failed_drunks.append(d)
                    movement = [0, 0]
                    break
                else:
                    direction = random.choice(checked)
                if direction == 0:
                    # Check north is valid. (Y)
                    if (drunk_y-1 >= 0):
                        if fallover or (walk_map[drunk_y-1][drunk_x] == None):
                            movement = [0, -1]
                        else:
                            checked.remove(direction)
                    else:
                        checked.remove(direction)
                elif direction == 1:
                    # Check east is valid. (X)
                    if (drunk_x+1 < len(voronoi[0])):
                        if fallover or (walk_map[drunk_y][drunk_x+1] == None):
                            movement = [1, 0]
                        else:
                            checked.remove(direction)
                    else:
                        checked.remove(direction)
                elif direction == 2:
                    # Check south is valid. (Y)
                    if (drunk_y+1 < len(voronoi)):
                        if fallover or (walk_map[drunk_y+1][drunk_x] == None):
                            movement = [0, 1]
                        else:
                            checked.remove(direction)
                    else:
                        checked.remove(direction)
                elif direction == 3:
                    # Check west is valid. (X)
                    if (drunk_x-1 >= 0):
                        if fallover or (walk_map[drunk_y][drunk_x-1] == None):
                            movement = [-1, 0]
                        else:
                            checked.remove(direction)
                    else:
                        checked.remove(direction)
            # Change position of drunk according to movement.
            drunks[d] = [drunks[d][0]+movement[0], drunks[d][1]+movement[1]]
        if debug:
            # Print steps.
            os.system("cls")
            for r, row in enumerate(walk_map):
                for c, column in enumerate(row):
                    if [r, c] in drunks: print(".", end="")
                    elif column == None: print(" ", end="")
                    else: print("#", end="")
                print()
            time.sleep(.1)
    # Finds all voronoi nodes types.
    voronoi_nodes = set([j for sub in walk_map for j in sub])
    if None in voronoi_nodes: voronoi_nodes.remove(None) # If the entire board is filled (BAD), we don't need to remove None
    return [[column if column in voronoi_nodes else None for column in row] for row in voronoi] # Returns a board of voronoi polygons and Nones.

def cellular_automata(board, rule, iters=1):
    for _ in range(iters):
        new_board = [] # So we don't have bias, and all effects happen indiscriminately
        for r, row in enumerate(board):
            new_board.append([]) # Th
            for c, column in enumerate(row):
                neighbours = []
                if r-1 >= 0:
                    if c-1 >= 0: neighbours.append(board[r-1][c-1])
                    if c+1 < len(board[0]): neighbours.append(board[r-1][c+1])
                    neighbours.append(board[r-1][c])
                if r+1 < len(board):
                    if c-1 >= 0: neighbours.append(board[r+1][c-1])
                    if c+1 < len(board[0]): neighbours.append(board[r+1][c+1])
                    neighbours.append(board[r+1][c])
                if c-1 >= 0: neighbours.append(board[r][c-1])
                if c+1 < len(board[0]): neighbours.append(board[r][c+1])
                if set(neighbours) != {None}: percentage = len(neighbours)/len([i for i  in neighbours if i != None])
                else:
                    percentage = 0
                if column == None:
                    new_board[-1].append(None)
                else:
                    if percentage > rule:
                        new_board[-1].append(1)
                    else:
                        new_board[-1].append(None)
        board = copy.deepcopy(new_board)
    return board

def poisson_disk(size_x, size_y, r, max_points):
    points_placed = set()
    prev_len = 0
    # Make a board with the correct dimensions
    for point in range(max_points):
        iteration = 0
        while (prev_len == len(points_placed)) and (iteration <= 100):
            iteration += 1
            x, y = random.randint(0, size_x-1), random.randint(0, size_y-1)
            if any([math.hypot(x-x1, y-y1) < r for x1, y1 in points_placed]): continue
            points_placed.add((x, y))
            prev_len = len(points_placed)
        if iteration == 100:
            break
    return points_placed

def generate_smoothing(points, run, terrain):
    board = []
    for r, row in enumerate(terrain):
        board.append([])
        for c, column in enumerate(row):
            possible_locations = list([gradient*(run-math.hypot(location[0]-c, location[1]-r)) for location, height, gradient in points if math.hypot(location[0]-c, location[1]-r) <= run])
            if possible_locations == []: value = 0
            else: value = round(sum(possible_locations)/len(possible_locations))
            if terrain:
                if terrain[r][c] != None and value < 0: value = 0
                elif terrain[r][c] == None and value > 0: value = 0
            board[-1].append(value)
    return board

def biome_select(terrain, height, temp, biomes, *, terrain_types=None):
    new_board = []
    for r, row in enumerate(terrain):
        new_board.append([])
        # New row
        for c, column in enumerate(row):
            biome_closest = ""
            found = False
            # some clamping here bc i can't help myself
            for biome in biomes:
                for gen in biomes[biome]["generation"]:
                    # We just need to know if it falls in my lovely little boxes :)
                    temp_gen = clamp(round(temp[r][c]), 0, 10) in gen["temp"]
                    height_gen = clamp(round(height[r][c]), -10, 10) in gen["elevation"]
                    #print(temp_gen and height_gen)
                    # Seperated for my convenience
                    if temp_gen or height_gen: 
                        found = True
                        biome_closest = biome
            """
            if not found:
                print("height", height[r][c], "temp", temp[r][c])
            new_board[-1].append(biomes[biome]["ID"])
            """
    return new_board

def in_range(n, s, l):
    return (n >= s) and (n <= l)

def generate_world(size_x, size_y, polygons=None, steps=None, drunks=None, temp_variance=None, poisson_r=None, poisson_max_points=None, height_weight=3, run=20, global_variance=50):
    
    print("Generating voronoi polygons", end="")
    time1 = time.time()
    # Step 0: set all options
    if polygons == None:
        polygons = round((size_x*size_y)*(1/5))
        if polygons == 0: polygons = 1
    if drunks == None:
        drunks = round((size_x*size_y)*(1/200))
        if drunks == 0: drunks = 1
    if steps == None:
        steps = round((size_x*size_y)*(1/100))
        if steps == 0: steps = 1
    if temp_variance == None:
        temp_variance = round((size_x*size_y)*(1/75))
        if temp_variance == 0: temp_variance = 1
    if poisson_r == None:
        poisson_r = 2+round(1+1/size_x+1/size_y)
    if poisson_max_points == None:
        poisson_max_points = round((size_x*size_y)*(1/200))
        if poisson_max_points == 0: poisson_max_points = 1
    # Step 1, generate voronoi to specifications.
    #         - Terrain: shape of coastlines etc. Small polygons
    terrain = generate_diagram(size_x, size_y, polygons)
    print(f"\rFound voronoi polygons ({round(time.time()-time1, 2)}s)")
    # Step 2, run a random walk algorithm over the voronoi. Only include polygons included in the walk.
    print("Generating drunken path", end="")
    time1 = time.time()
    terrain = random_walk(terrain, steps, drunks, fallover=False)
    # Step 3, run cellular automata to change cells
    print(f"\rCreated drunken path ({round(time.time()-time1, 2)}s)")
    print("Applying cellular automata", end="")
    time1 = time.time()
    terrain = cellular_automata(terrain, 0.5, iters=2)
    print(f"\rApplied cellular automata ({round(time.time()-time1, 2)}s)\n")
    """
    print("Generating height and temperature points.", end="")
    time1 = time.time()
    # Step 4, generate height map, then generate terrain
    height_points = list(poisson_disk(size_x, size_y, poisson_r, poisson_max_points))
    temp = list(poisson_disk(size_x, size_y, poisson_r, poisson_max_points))
    print(f"\rCreated height and temp point maps. ({round(time.time()-time1, 2)}s)")
    print("Generating height and temp graphs", end="")
    time1 = time.time()
    points_values = []
    # create a weighted set of values as heights
    for point in height_points:
        value = round(sum([random.random()*100 for i in range(height_weight)])/height_weight, 1)
        # everything underwater is negative (0.0 is sea level)
        if terrain[point[1]][point[0]] == None:
            value *= -1
        # Append the location, value, and gradient.
        points_values.append((point, value, value/run))
    # Generate the temperature now
    height_map = generate_smoothing(points_values, run, terrain)
    noise = PerlinNoise(octaves=10)
    temp_map = [[round(clamp(abs(noise([i/size_x+.5, j/size_y+.5])*global_variance), 0, 10), 1) for j in range(size_x)] for i in range(size_y)]
    print(f"\rCreated height and temperature maps. ({round(time.time()-time1, 2)}s)")
    """
    #print(terrain)
    return terrain

def find_continents(world):
    board = copy.deepcopy(world)
    cont_names = markov.generate_chain(open("continent_names.txt").read().split("\n"))
    stack = []
    max_x, max_y = len(board[0])-1, len(board)-1
    continents = {}
    board_size = len([j for sub in board for j in sub])
    continent = 0
    while any([i==1 for i in [j for sub in board for j in sub]]):
        name = markov.generate_result(cont_names, max_length=7)
        while name in continents.keys():
            name = markov.generate_result(cont_names, max_length=7)
        time1 = time.time()
        size = 0
        for r_o, row in enumerate(board):
            for c_o, column in enumerate(row):
                if column == 1: break
            if column == 1: break
        stack.append((r_o, c_o))
        continents[name] = []
        stack_length = 0
        while len(stack) > 0:
            size += 1
            r, c = stack[len(stack)-1]
            del stack[-1]
            stack_loc = len(stack)
            board[r][c] = None
            if r-1 >= 0:
                if board[r-1][c] != None and (r, c) not in stack: stack.append((r-1, c))
            if r+1 <= max_x:
                if board[r+1][c] != None and (r, c) not in stack: stack.append((r+1, c))
            if c-1 >= 0:
                if board[r][c-1] != None and (r, c) not in stack: stack.append((r, c-1))
            if c+1 <= max_y:
                if board[r][c+1] != None and (r, c) not in stack: stack.append((r, c+1))
            if len(stack) > stack_length: stack_length = len(stack) 
            continents[name].append((c, r))
            #time.sleep(0.001)
            #print(f"\rContinent {continent}: {stack_loc}/{size}", end="")
        print(f"\rContinent {continent} ({round(time.time()-time1, 2)}s): {stack_loc}/{size}")
        continent += 1
    return continents

def determine_nation_knowledge(continents, nations):
    continent_knowledge = {key:[] for key in continents.keys()}
    personal_knowledge = {}
    knowledge = {key:[] for key in nations.keys()}
    # Now, time to fill them
    for nation in nations:
        for city in nations[nation]["cities"]:
            continents_known = [continent for continent in continents if city["location"] in continents[continent]]
            personal_knowledge[nation] = continents_known
            for i in continents_known: continent_knowledge[i].append(nation)
    for nation in personal_knowledge:
        for i in [continent_knowledge[i] for i in personal_knowledge[nation]]:
            knowledge[nation] += i
    knowledge = {key:set([i for i in value if i != key]) for key, value in knowledge.items()}
    return knowledge, personal_knowledge

def generate_unique_colours(colours, dist, luma_min=75):
    colours_ = []
    for _ in range(colours):
        x, y, z = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
        luma = 0.2126 * x + 0.7152 * y + 0.0722 * z
        if len(colours_) != 0:
            while any([math.hypot(x, y, z) < dist for x1, y1, z1 in colours_]):
                x, y, z = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
                if luma < luma_min: continue
        colours_.append((x, y, z))
    return colours_

def generate_civs(world, civ_no, civ_distance=7):
    # Step 1: create names
    print("Generating civilisations", end="")
    time1 = time.time()
    print(" . Generating markov chains")
    town_names = markov.generate_chain(open("town_names.txt").read().split("\n"))
    nation_names = markov.generate_chain(open("nation_names.txt").read().split("\n"))
    civs = {}
    for civ in range(civ_no):
        name = markov.generate_result(nation_names, max_length=10)
        while name in civs.keys(): name = markov.generate_result(town_names, min_length=5, max_length=10)
        civs[name] = {}
    # Step 2: generate civ positions
    world_x, world_y = len(world[0]), len(world)
    civ_locs = []
    # Create a whole bunch of locations
    for civ in civs.keys():
        # Implement a poisson disk-like equation.
        unsuitable = True
        iter_ = 0
        while unsuitable and iter_ < 100:
            iter_ += 1
            # Repeat until finding a suitable location.
            civ_x, civ_y = random.randint(0, world_x-1), random.randint(0, world_y-1)
            if (civ_x, civ_y) in civ_locs:
                # Can't put two cities on top each other. Techincally.
                unsuitable = True
            elif world[civ_y][civ_x] == None:
                # Well I mean cities don't just START floating
                unsuitable = True
            elif all([math.hypot(civ_x-x, civ_y-y) > civ_distance for x, y in civ_locs]) or len(civ_locs)==0:
                # Ok... I don't want them too far away, so...
                unsuitable = False
        if unsuitable:
            break
        else:
            civ_locs.append((civ_x, civ_y))
            civs[civ]["cities"] = [{"location": (civ_x, civ_y),
                                    "population": 3, 
                                    "defense": 50, 
                                    "military": 0,
                                    "mood": 10,
                                    "name": markov.generate_result(town_names, max_length=10),
                                    "tiles_owned": [(civ_x, civ_y)]}]
        civs[civ]["technology"] = 0
        civs[civ]["religion"] = {}
    print(f"\rGenerated civilisations. ({round(time.time()-time1, 2)}s)")
    # Remove all civilisations if they don't contain a location
    civs = {key:value for key, value in civs.items() if len(value.keys())>0}
    print("Finding world continents:")
    continents = find_continents(world)
    #print("Determining political structure.")
    politics_structure, home_continents = determine_nation_knowledge(continents, civs)
    civs_colours = generate_unique_colours(len(list(civs.keys())), 8)
    for n, nation in enumerate(politics_structure):
        civs[nation]["relations"] = {value:0 for value in politics_structure[nation]}
        civs[nation]["home_continent"] = home_continents[nation]
        civs[nation]["colour"] = civs_colours[n]
    return civs

def main():
    WORLD_SIZE = 200
    world_start = time.time()
    world = generate_world(WORLD_SIZE, WORLD_SIZE)
    print(f"World gen done. {round(time.time()-world_start, 2)}s")
    civs = generate_civs(world, round(WORLD_SIZE/5))
    text = ["Countries and their cities:"]
    for civ in {key:value for key, value in sorted(civs.items())}:
        text.append(f"\u001b[38;2;%s;%s;%sm {civ}\u001b[0m"%civs[civ]["colour"])
        for city in civs[civ]["cities"]:
            name = city["name"]
            text.append(f"\u001b[38;2;%s;%s;%sm . {name}\u001b[0m"%civs[civ]["colour"])
    try:
        city_locs = []
        for civ in civs:
            for city in civs[civ]["cities"]:
                city_locs += city["tiles_owned"]
        for r, row in enumerate(world):
            for c, column in enumerate(row):
                if (c, r) in city_locs:
                    for civ in civs:
                        for city in civs[civ]["cities"]:
                            if (c, r) in city["tiles_owned"]:
                                print("\u001b[38;2;%s;%s;%sm#\u001b[0m"%civs[civ]["colour"], end="")
                                break
                elif column == 1:
                    print(".", end="")
                else:
                    print(" ", end="")
            if r < len(text):
                print(text[r])
            else:
                print()
        while r+1 < len(text):
            r += 1
            print(" "*(WORLD_SIZE-1), text[r])
    except KeyboardInterrupt:
        print("Ending sim!")

if __name__ == "__main__": main()