from functools import wraps

from flask import flash, redirect, url_for
from flask_login import current_user, login_required


def admin_required(f):
    @wraps(f)
    @login_required
    def decorated(*args, **kwargs):
        if not current_user.is_admin:
            flash("Akses ditolak. Hanya admin yang bisa mengakses halaman ini.", "error")
            return redirect(url_for("dashboard"))
        return f(*args, **kwargs)

    return decorated