import bcrypt
import getpass

if __name__ == "__main__":
    try:
        pw = getpass.getpass("Enter the password to hash: ")
        hashed = bcrypt.hashpw(pw.encode("utf-8"), bcrypt.gensalt())
        print("\nAPP_PASSWORD_HASH=" + hashed.decode("utf-8"))
    except Exception as e:
        print(f"Error generating hash: {e}")
