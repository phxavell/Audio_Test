import numpy as np
import pyaudio
import wave
import librosa
import matplotlib.pyplot as plt
import time

# Configurações de áudio
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
DURATION = 5  # Duração da gravação em segundos

def record_audio(filename):
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Gravando...")
    frames = []

    for i in range(0, int(RATE / CHUNK * DURATION)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Gravação concluída.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def plot_waveforms(y_original, y_recorded, sr):
    plt.figure(figsize=(12, 6))

    # Plot forma de onda do áudio original
    plt.subplot(2, 1, 1)
    plt.plot(np.linspace(0, len(y_original) / sr, num=len(y_original)), y_original)
    plt.title('Forma de Onda - Áudio Original')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Amplitude')
    plt.grid()

    # Plot forma de onda do áudio gravado
    plt.subplot(2, 1, 2)
    plt.plot(np.linspace(0, len(y_recorded) / sr, num=len(y_recorded)), y_recorded)
    plt.title('Forma de Onda - Áudio Gravado')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Amplitude')
    plt.grid()

    plt.tight_layout()
    plt.show(block=False)  # Não bloquear a execução
    plt.pause(5)  # Esperar 2 segundos
    plt.close()  # Fechar a figura

def plot_spectra(y_original, y_recorded, sr):
    plt.figure(figsize=(12, 6))

    # Espectro do áudio original
    D_orig = np.abs(librosa.stft(y_original))
    plt.subplot(2, 1, 1)
    plt.imshow(librosa.amplitude_to_db(D_orig, ref=np.max), origin='lower', aspect='auto', 
               cmap='jet', extent=[0, len(y_original) / sr, 0, sr / 2])
    plt.title('Espectro - Áudio Original')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Frequência (Hz)')

    # Espectro do áudio gravado
    D_rec = np.abs(librosa.stft(y_recorded))
    plt.subplot(2, 1, 2)
    plt.imshow(librosa.amplitude_to_db(D_rec, ref=np.max), origin='lower', aspect='auto', 
               cmap='jet', extent=[0, len(y_recorded) / sr, 0, sr / 2])
    plt.title('Espectro - Áudio Gravado')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Frequência (Hz)')

    plt.tight_layout()
    plt.show(block=False)  # Não bloquear a execução
    plt.pause(5)  # Esperar 2 segundos
    plt.close()  # Fechar a figura

def analyze_audio(recorded_file, original_file, target_frequencies):
    try:
        y_recorded, sr_recorded = librosa.load(recorded_file, sr=None, dtype=np.float32)
        y_original, sr_original = librosa.load(original_file, sr=None, dtype=np.float32)

        # Normalizar os sinais
        y_recorded /= np.max(np.abs(y_recorded)) if np.max(np.abs(y_recorded)) != 0 else 1
        y_original /= np.max(np.abs(y_original)) if np.max(np.abs(y_original)) != 0 else 1

        # Garantir que ambos os sinais tenham o mesmo comprimento
        min_length = min(len(y_recorded), len(y_original))
        y_recorded = y_recorded[:min_length]
        y_original = y_original[:min_length]

        # Plotar formas de onda
        plot_waveforms(y_original, y_recorded, sr_original)

        # Plotar espectros
        plot_spectra(y_original, y_recorded, sr_original)

        # Análise de frequências
        D_orig = np.abs(librosa.stft(y_original))
        D_rec = np.abs(librosa.stft(y_recorded))

        # Média das amplitudes nas frequências alvo
        target_amplitudes_orig = np.mean(D_orig[target_frequencies, :], axis=0)
        target_amplitudes_rec = np.mean(D_rec[target_frequencies, :], axis=0)

        # Comparar as amplitudes
        similarity = np.corrcoef(target_amplitudes_orig, target_amplitudes_rec)[0, 1]

        print(f"Similaridade nas frequências alvo: {similarity:.2f}")

        # Definir limite de aceitação para similaridade
        threshold = 0.7  # Ajustar conforme necessário

        if similarity >= threshold:
            print("Teste aprovado!")
        else:
            print("Teste reprovado!")

    except Exception as e:
        print(f"Ocorreu um erro ao analisar o áudio: {e}")

if __name__ == "__main__":
    original_audio_file = 'som_original.wav'
    recorded_audio_file = 'som_gravado.wav'

    # Defina as frequências alvo (exemplo: 100 Hz e 200 Hz)
    target_frequencies = [19500,21000]  # Substitua pelos índices correspondentes no espectro

    record_audio(recorded_audio_file)

    analyze_audio(recorded_audio_file, original_audio_file, target_frequencies)
