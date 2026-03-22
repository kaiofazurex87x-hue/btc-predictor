import bcrypt

print("=" * 40)
pw     = input("Choose a password: ")
hashed = bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()
print("\nCopy this as your ADMIN_PASSWORD_HASH in Render:")
print(hashed)
