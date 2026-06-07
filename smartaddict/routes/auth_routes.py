from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user

from smartaddict.extensions import db
from smartaddict.models.user import User

auth_bp = Blueprint('auth', __name__)


@auth_bp.route("/", endpoint='index')
def index():
    return redirect(url_for("user.dashboard"))


@auth_bp.route("/login", methods=["GET", "POST"], endpoint='login')
def login():
    if current_user.is_authenticated:
        return redirect(url_for("user.dashboard"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            flash(f"Selamat datang, {user.username}! =ƒæï", "success")
            next_page = request.args.get("next")
            if user.is_admin:
                return redirect(next_page or url_for("admin.admin_dashboard"))
            return redirect(next_page or url_for("user.dashboard"))
        flash("Username atau password salah.", "error")
    return render_template("auth/login.html")


@auth_bp.route("/register", methods=["GET", "POST"], endpoint='register')
def register():
    if current_user.is_authenticated:
        return redirect(url_for("user.dashboard"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")
        
        # Validation without flash messages
        if len(username) >= 3 and len(password) >= 6 and password == confirm:
            if not User.query.filter_by(username=username).first():
                user = User(username=username, role="user")
                user.set_password(password)
                db.session.add(user)
                db.session.commit()
                login_user(user)
                flash("Registrasi berhasil! Selamat datang! =ƒÄë", "success")
                return redirect(url_for("user.dashboard"))
        # Removed all validation error flash messages
    return render_template("auth/register.html")


@auth_bp.route("/logout", endpoint='logout')
@login_required
def logout():
    logout_user()
    flash("Berhasil logout.", "success")
    return redirect(url_for("auth.login"))
