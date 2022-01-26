# Generate Markov
# coding=utf8

import pprint
from random import choices

def generate_chain(input, word_chars=False):
    chars = {}
    if word_chars:
        input = [sample.split() for sample in input]
    for sample in input:
        for c, character in enumerate(sample):
            # Stop character.
            if c == 0:
                if "START" in chars.keys():
                    if character in chars["START"].keys(): chars["START"][character] += 1
                    else: chars["START"][character] = 1
                else:
                    chars["START"] = {character: 1}
            if c == len(sample)-1:
                if character in chars.keys():
                    if "STOP" in chars[character].keys(): chars[character]["STOP"] += 1
                    else: chars[character]["STOP"] = 1
                else:
                    chars[character] = {"STOP": 1}
            else:
                if character in chars.keys():
                    if sample[c+1] in chars[character].keys(): chars[character][sample[c+1]] += 1
                    else: chars[character][sample[c+1]] = 1
                else:
                    chars[character] = {sample[c+1]: 1}
    return chars

def generate_result(chain, max_length=15, min_length=3, word_chars=False):
    text = []
    while len(text) < min_length or len(text) > max_length:
        character = "START"
        text = []
        while character != "STOP":
            #print(chain[character])
            characters = chain[character]
            char = choices(
                list(characters.keys()),
                weights=list(characters.values())
            )[0]
            text.append(char)
            character = char
        text.remove("STOP")
    if word_chars:
        return " ".join(text)
    return "".join(text)