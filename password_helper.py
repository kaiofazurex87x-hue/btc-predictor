import bcrypt
print("=" * 40)
pw     = input("Choose a password: ")
hashed = bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()
print("\nSet this as your ADMIN_PASSWORD_HASH environment variable in PythonAnywhere:")
print(hashed)
