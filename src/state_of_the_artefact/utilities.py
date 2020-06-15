def reverse_sequences(sequences):
    return [seq[::-1] for seq in sequences]


def make_artefact(agent, artefact):
    return {
        agent: agent.id,
        culture: agent.culture_id,
        artefact: artefact
    }