from typing import Dict, List

registered_generators: Dict[str, 'Generator'] = {}
registered_groups: Dict[str, List['Generator']] = {}


def register_generator(gen: 'Generator', name: str):
    registered_generators[name] = gen


def register_group(gen: 'Generator', group: str):
    print(group)
    if group not in registered_groups:
        registered_groups[group] = []

    registered_groups[group].append(gen)


def get_generator_by_name(name: str) -> 'Generator':
    if name not in registered_generators:
        raise KeyError("Could not find generator with name {}".format(name))

    return registered_generators[name]


def get_group_by_name(group: str) -> List['Generator']:
    if group not in registered_groups:
        raise KeyError("Could not find generator group with name {}".format(group))

    return registered_groups[group]
