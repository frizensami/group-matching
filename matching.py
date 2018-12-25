import itertools

PARTICIPANTS_COUNT = 30
GROUP_SIZE = 3
NUM_GROUPS = int(PARTICIPANTS_COUNT / GROUP_SIZE)

participants = [i for i in range(PARTICIPANTS_COUNT)]
rankings = [[x+3 for x in range(PARTICIPANTS_COUNT)] for y in range(PARTICIPANTS_COUNT)]
for i in range(PARTICIPANTS_COUNT):
    rankings[i][i] = 0 # cannot rank yourself

print(participants)
print(rankings)


def partition(collection):
    if len(collection) == 1:
        yield [ collection ]
        return

    first = collection[0]
    for smaller in partition(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[ first ] + subset]  + smaller[n+1:]
        # put `first` in its own subset 
        yield [ [ first ] ] + smaller

partitions = partition(participants)
#for p in partitions:
#    print(p)


def part(agents, items):
    if len(agents) == 1:
        yield {agents[0]: items}
    else:
        quota = len(items) // len(agents)
        for indexes in combinations(range(len(items)), quota):
            remainder = items[:]
            selection = [remainder.pop(i) for i in reversed(indexes)][::-1]
            for result in part(agents[1:], remainder):
                result[agents[0]] = selection
                yield result

def get_group_score(group, rankings):
    total = 0
    for cur in group:
        for other in group:
            total += rankings[cur][other]
    
    return total
'''
# (x1 x2 x3), (y11, y2, y3) .... 
groups = list(itertools.combinations(participants, GROUP_SIZE))
group_scores = []
for g in groups:
    score = get_group_score(g, rankings)
    #print(str(g) + ": " + str(score))
    group_scores.append((g, score))

group_scores.sort(key=lambda x: x[1], reverse=True)

for g in group_scores:
    print(g)
'''
# Problem is that choosing 10 groups from this combination of 4060 groups is insane


# Find all partitions that have sets of all size GROUP_SIZE
#partitions_of_size = filter(lambda p: all(map(lambda v: len(v) == GROUP_SIZE, p)), partitions)

#for p in partitions_of_size:
#	print("Partition: ", p)
#	print("")

