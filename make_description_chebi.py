import obonet
import json
import obonet
import networkx
import fastobo
import ast
import pickle

def find_child_chebi(chebi_list):
    chebi_dag = {}
    for che in chebi_list:
        chebi_dag[che] = []
    knowledge_graph = networkx.DiGraph()
    #read obo file
    pato = fastobo.load('./Ref_files/chebi_core.obo')

    # populate the knowledge graph with is_a relationships
    for frame in pato:
        if isinstance(frame, fastobo.term.TermFrame):
            knowledge_graph.add_node(str(frame.id))
            for clause in frame:
                if isinstance(clause, fastobo.term.IsAClause):
                    knowledge_graph.add_edge(str(frame.id), str(clause.term))

    # find the leaves from all of the nodes
    nodes = knowledge_graph.nodes

    for chebi in chebi_list:
      chebi_dag[chebi] = list(networkx.ancestors(knowledge_graph, chebi))


    #with open('Mid_files/dict_CHEBI_Dag_Ontoclass.txt', 'w') as data:
     #   data.write(str(chebi_dag))

    return chebi_dag


def get_chebi_smiles(chebi_list):

    graph = obonet.read_obo('./Ref_files/chebi_core.obo')
    smiles = {}
    for chebi in chebi_list:
        smiles[chebi] = []
    for chebi in chebi_list:
        if chebi in graph.nodes.keys():
           if 'property_value' in list(graph.nodes[chebi].keys()):
              for item in graph.nodes[chebi]['property_value']:
                  item = item.split(' ')
                  if 'smiles' in item[0]:
                      smiles[chebi].append(item[1])
           else:
               chebi_child = find_child_chebi([chebi])
               print(chebi_child)
               print(chebi_child[chebi])
               for child in chebi_child[chebi]:
                   print(graph.nodes[child].keys())
                   print(child)
                   if 'property_value' in list(graph.nodes[child].keys()):
                       for item in graph.nodes[child]['property_value']:
                           item = item.split(' ')
                           if 'smiles' in item[0]:
                              smiles[chebi].append(item[1].strip('"'))
                           else:
                              smiles[chebi].append('None')
    return smiles
def get_alt_ids():
    alt_ids = {}
    graph = obonet.read_obo('./Ref_files/chebi_core.obo')
    for item in list(graph.nodes.keys()):
        if 'alt_id' in graph.nodes[item]:
           alt_ids[item] = graph.nodes[item]['alt_id']
    json.dump(alt_ids, open("./Dataset/spot/alt_ids.txt", 'w'))
    return alt_ids

def get_chebi_des(chebi_list):
    '''

    :param chebi_list: a list of chebi terms
    :return: a dictionary of chebi terms and their corresponding descriptions 
    :return: a dictionary of chebi terms and their corresponding smiles
    '''
    graph = obonet.read_obo('./Ref_files/chebi_core.obo')
    desc = {}
    for chebi in chebi_list:
        print(chebi)
        if chebi in graph.nodes.keys():
           print(list(graph.nodes[chebi].keys()))
           if 'def' in list(graph.nodes[chebi].keys()):
              desc[chebi] = graph.nodes[chebi]['name']+': '+graph.nodes[chebi]['def'].strip('[]')


           else:
              desc[chebi] = graph.nodes[chebi]['name']

    return desc
   

def find_keys_by_value(search_string, dictionary):
    # Return a list of all keys where the search_string is found in the value list
    return [key for key, value_list in dictionary.items() if search_string in value_list]

def get_chebi_name(chebi_list):
    '''

    :param chebi_list: a list of chebi terms
    :return: a dictionary of chebi terms and their corresponding descriptions 
    :return: a dictionary of chebi terms and their corresponding smiles
    '''
    graph = obonet.read_obo('./Ref_files/chebi_core.obo')
    name = {}
    for chebi in chebi_list:
        if chebi in graph.nodes.keys():
           name[chebi] = graph.nodes[chebi]['name']


    return name


def update_term_des_with_alt_ids(chebi_list, term_des, alt_ids):
    '''
    Updates term descriptions using alternative IDs.

    :param chebi_list: the original list of chebi terms
    :param term_des: the initial dictionary of chebi descriptions
    :param alt_ids: a dictionary mapping chebi terms to alternative IDs
    :return: an updated term_des dictionary with missing terms filled using alternative IDs
    '''
    # Identify missing terms from the original chebi_list
    missing_des_ids = [item for item in chebi_list if item not in term_des]

    # Get alternative IDs for missing terms
    alt_ids_reverse = []
    alt_ids_reverse_dict = {}
    for item in missing_des_ids:
        alt = find_keys_by_value(item, alt_ids)[0]
        alt_ids_reverse.append(alt)  # Add alternative IDs for each missing item
        alt_ids_reverse_dict[item] = alt
    # Fetch descriptions for the alternative IDs
    alt_des = get_chebi_des(alt_ids_reverse)

    # Map descriptions from alternative IDs back to the original missing IDs
    for item in missing_des_ids:
        if alt_ids_reverse_dict[item] in alt_des.keys():
            term_des[item] = alt_des[alt_ids_reverse_dict[item]]  # Add description using original missing ID


    return term_des

def update_term_smiles_with_alt_ids(chebi_list, term_smiles, alt_ids):
    '''
    Updates term descriptions using alternative IDs.

    :param chebi_list: the original list of chebi terms
    :param term_des: the initial dictionary of chebi descriptions
    :param alt_ids: a dictionary mapping chebi terms to alternative IDs
    :return: an updated term_des dictionary with missing terms filled using alternative IDs
    '''
    # Identify missing terms from the original chebi_list
    missing_des_ids = [item for item in chebi_list if item not in term_smiles]

    # Get alternative IDs for missing terms
    alt_ids_reverse = []
    alt_ids_reverse_dict = {}
    for item in missing_des_ids:
        alt = find_keys_by_value(item, alt_ids)[0]
        alt_ids_reverse.append(alt)  # Add alternative IDs for each missing item
        alt_ids_reverse_dict[item] = alt
    # Fetch descriptions for the alternative IDs
    alt_des = get_chebi_smiles(alt_ids_reverse)

    # Map descriptions from alternative IDs back to the original missing IDs
    for item in missing_des_ids:
        if alt_ids_reverse_dict[item] in alt_des.keys():
            term_smiles[item] = alt_des[alt_ids_reverse_dict[item]]  # Add description using original missing ID


    return term_smiles

def update_term_name_with_alt_ids(chebi_list, term_name, alt_ids):
    '''
    Updates term descriptions using alternative IDs.

    :param chebi_list: the original list of chebi terms
    :param term_des: the initial dictionary of chebi descriptions
    :param alt_ids: a dictionary mapping chebi terms to alternative IDs
    :return: an updated term_des dictionary with missing terms filled using alternative IDs
    '''
    # Identify missing terms from the original chebi_list
    missing_des_ids = [item for item in chebi_list if item not in term_name]

    # Get alternative IDs for missing terms
    alt_ids_reverse = []
    alt_ids_reverse_dict = {}
    for item in missing_des_ids:
        alt = find_keys_by_value(item, alt_ids)[0]
        alt_ids_reverse.append(alt)  # Add alternative IDs for each missing item
        alt_ids_reverse_dict[item] = alt
    # Fetch descriptions for the alternative IDs
    alt_des = get_chebi_name(alt_ids_reverse)

    # Map descriptions from alternative IDs back to the original missing IDs
    for item in missing_des_ids:
        if alt_ids_reverse_dict[item] in alt_des.keys():
            term_name[item] = alt_des[alt_ids_reverse_dict[item]]  # Add description using original missing ID


    return term_name



chebi_list = []
#file_path = './Dataset/spot/chebi_terms.txt'
#file_path ='./Ref_files/chebi_leaf.txt'
term_number ={}
#file_path = './Dataset/ICAT/Label_name_list_ICAT_uni_ident100_t10'
file_path = './Dataset/Label_name_list_transporter_uni_ident100_D_minority'
with open(file_path, 'r') as file:
    for line in file:
        columns = line.strip().split(',')
        chebi_list.append(columns[0])
        term_number[columns[0]] = columns[2]

file_path = './Dataset/Label_name_list_transporter_uni_ident100_t3'
with open(file_path, 'r') as file:
    for line in file:
        columns = line.strip().split(',')
        chebi_list.append(columns[0])
        term_number[columns[0]] = columns[2]
alt_ids = get_alt_ids()
a=1


term_des = get_chebi_des(chebi_list)
term_smiles = get_chebi_smiles(chebi_list)
term_name = get_chebi_name(chebi_list)

term_smiles = update_term_smiles_with_alt_ids(chebi_list,term_smiles, alt_ids)
term_des = update_term_des_with_alt_ids(chebi_list, term_des, alt_ids)
term_name = update_term_name_with_alt_ids(chebi_list, term_name, alt_ids)

json.dump(term_des, open("./Dataset/chebi_des_D_minority_and_t3.txt",'w'))
json.dump(term_smiles, open("./Dataset/chebi_smiles_D_minority_and_t3.txt",'w'))
json.dump(term_name, open("./Dataset/chebi_name_D_minority_and_t3.txt",'w'))


number_smiles = {}
number_des = {}
number_name = {}

#with open('./Dataset/ICAT/chebi_index_dict.json', 'rb') as file:
#    term_number = json.load(file)

number_des = {term_number[term]:term_des[term] for term in term_number if term in term_des}
json.dump(number_des, open("./Dataset/number_des_D_minority_and_t3.txt",'w'))

number_smiles = {term_number[term]:term_smiles[term] for term in term_number if term in term_smiles}
json.dump(number_smiles, open("./Dataset/number_smiles_D_minority_and_t3.txt",'w'))

number_name = {term_number[term]:term_name[term] for term in term_number if term in term_name}
json.dump(number_name, open("./Dataset/number_name_D_minority_and_t3.txt",'w'))

print(number_des)
print(number_smiles)

