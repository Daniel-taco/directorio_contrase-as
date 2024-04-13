import itertools
import time


def generate_passwords(length):
    characters = 'abcdefghijklmnñopqrstuvwxyz0123456789'
    passwords = [''.join(password) for password in itertools.product(characters, repeat=length)]
    return passwords

def save_passwords_to_file(passwords):
    with open('contraseñas.txt', 'w') as file:
        for password in passwords:
            file.write(password + '\n')

def main():
    length = int(input("Ingrese la longitud de la contraseña: "))
    start_time = time.time()
    passwords = generate_passwords(length)
    end_time = time.time()
    duration = end_time - start_time
    print("Se generó en ", duration)
    save_passwords_to_file(passwords)
    print(f"Se han generado y guardado {len(passwords)} contraseñas en 'contraseñas.txt'.")

if __name__ == "__main__":
    main()
