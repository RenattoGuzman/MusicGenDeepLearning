import os
import numpy as np
from music21 import converter, instrument, note, chord, stream
from keras.models import load_model


# Función para procesar los archivos MIDI
def get_notes():
    notes = []
    for file in os.listdir('MJ'):
        if file.endswith(".mid"):
            midi = converter.parse(os.path.join('MJ', file))
            notes_to_parse = None
            try:
                s2 = instrument.partitionByInstrument(midi)
                notes_to_parse = s2.parts[0].recurse()
            except:
                notes_to_parse = midi.flat.notes
            
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
    
    return notes


# Generar música
def generate_notes(model, network_input, pitchnames, n_vocab):
    start = np.random.randint(0, len(network_input)-1)
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    pattern = network_input[start]
    prediction_output = []

    for _ in range(500):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)
        prediction = model.predict(prediction_input, verbose=0)
        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)
        pattern = np.append(pattern, index)
        pattern = pattern[1:len(pattern)]

    return prediction_output


# Nueva función para crear el archivo MIDI
def create_midi(prediction_output, filename="output.mid"):
    offset = 2
    output_notes = []

    # Crear objetos note y chord basados en los valores generados
    for pattern in prediction_output:
        # Si el patrón es un acorde
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # Si el patrón es una nota
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # Incrementar el offset para que las notas no se superpongan
        offset += 0.5

    # Crear un stream de música21
    midi_stream = stream.Stream(output_notes)

    # Escribir el stream en un archivo MIDI
    midi_stream.write('midi', fp=filename)

    midi_stream.show('midi')


def load_trained_model(model_path):
    # Find the most recent model directory
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    # Load the model
    model = load_model(model_path)
    print(f"Model loaded successfully from {model_path}")
    return model


def main():
    
    # Preprocesamiento
    notes = get_notes()
    n_vocab = len(set(notes))
    note_to_int = dict((note, number) for number, note in enumerate(set(notes)))

    sequence_length = 100
    network_input = []
    network_output = []

    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input = network_input / float(n_vocab)

    print("Preprocesamiento completado")

    # Código para generar y guardar nueva música
    loaded_model = load_trained_model("model/final_model.h5")
    prediction_output = generate_notes(loaded_model, network_input, set(notes), n_vocab)
    
    create_midi(prediction_output, "output3.mid")

    print("Archivo MIDI generado")


if __name__ == '__main__':
    main()
    print("Song generated successfully")