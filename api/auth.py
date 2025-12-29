from fastapi import HTTPException, status
from jose import JWTError, jwt
from api.database import SECRET_KEY, ALGORITHM, verify_pass, create_token

# Token verify karne ka function (Security ke liye)
def verify_access_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Token")
        return email
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials")