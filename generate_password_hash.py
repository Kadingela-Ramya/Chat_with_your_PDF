import bcrypt
import getpass

pw = getpass.getpass("Enter the password to hash: ")
hashed = bcrypt.hashpw(pw.encode("utf-8"), bcrypt.gensalt())
print("\nAPP_PASSWORD_HASH=" + hashed.decode("utf-8"))