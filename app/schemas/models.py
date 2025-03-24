# app/schemas/models.py
# defines the ORM
from sqlalchemy import Column, Integer, String, Date, DateTime, Numeric, Boolean, Text, ForeignKey
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True)
    password = Column(String(255))
    activated = Column(Boolean, default=False)
    activated_at = Column(DateTime)
    last_login = Column(DateTime)
    reset_password_code = Column(String(255))
    first_name = Column(String(255))
    last_name = Column(String(255))
    deleted_at = Column(DateTime)
    country = Column(String(50))
    avatar = Column(String(255))
    department_id = Column(Integer, ForeignKey('department.id'))
    location_id = Column(Integer, ForeignKey('location.id'))
    phone = Column(String(20))
    jobtitle = Column(String(255))
    manager_id = Column(Integer, ForeignKey('users.id'))
    employee_num = Column(String(20))
    username = Column(String(255))
    show_in_list = Column(Boolean, default=True)
    two_factor_secret = Column(String(255))
    two_factor_recovery_codes = Column(Text)
    two_factor_enabled = Column(Boolean, default=False)
    two_factor_opt_in = Column(Boolean, default=False)
    address = Column(String(255))
    city = Column(String(255))
    state = Column(String(255))
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    remember_token = Column(String(100))
    email_verified_at = Column(DateTime)

    #  Relationships
    department = relationship("Department", backref=backref("users", uselist=True), foreign_keys=[department_id])
    location = relationship("Location", back_populates="users", foreign_keys=[location_id])
    manager = relationship("User", remote_side=[id], back_populates="subordinates")
    subordinates = relationship("User", back_populates="manager", foreign_keys=[manager_id])
    # Relationships updated with string-based foreign_keys
    managed_assets = relationship("Asset", back_populates="manager", foreign_keys="Asset.manager_id")
    owned_assets = relationship("Asset", back_populates="user", foreign_keys="Asset.user_id")
    assigned_assets = relationship("Asset", back_populates="assigned_to", foreign_keys="Asset.assigned_user")
    requested_assets = relationship("Asset", back_populates="requesting", foreign_keys="Asset.requesting_user")


class Department(Base):
    __tablename__ = 'department'
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'))
    location_id = Column(Integer, ForeignKey('location.id'))
    manager_id = Column(Integer, ForeignKey('users.id'))
    notes = Column(Text)
    image = Column(Text)
    
    # Relationships
    location = relationship("Location", back_populates="departments")
    head = relationship("User", foreign_keys=[user_id], backref="headed_departments")
    manager = relationship("User", foreign_keys=[manager_id], backref="managed_departments")
    assets = relationship("Asset", back_populates="department")  # Added this line
    # Note: 'users' is provided by backref in User.department

class Location(Base):
    __tablename__ = 'location'
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    city = Column(String(255), nullable=False)
    state = Column(String(255), nullable=False)
    country = Column(String(255), nullable=False)
    address = Column(String(255), nullable=False)
    address2 = Column(String(255))
    postcode = Column(String(255), nullable=False)
    manager_id = Column(Integer, ForeignKey('users.id'))
    image = Column(Text)
    latitude = Column(Numeric(10,8))
    longitude = Column(Numeric(11,8))

    # relationships
    departments = relationship("Department", back_populates="location")
    manager = relationship("User", foreign_keys=[manager_id], backref="managed_locations")
    users = relationship("User", back_populates="location", foreign_keys=[User.location_id])
    assets = relationship("Asset", back_populates="location")

class Status(Base):
    __tablename__ = 'status'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    deployable = Column(Boolean, nullable=False)
    pending = Column(Boolean, nullable=False)
    archived = Column(Boolean, nullable=False)
    notes = Column(Text)
    color = Column(Text, nullable=False)
    show_in_nav = Column(Boolean, nullable=False)
    name = Column(String(255))

    user = relationship("User", backref="statuses")
    assets = relationship("Asset", back_populates="status")

# class Asset(Base):
#     # specifies the name of the table in the database
#     __tablename__ = 'assets'

#     id = Column(Integer, primary_key=True, index=True)
#     name = Column(String(255), index=True)
#     asset_tag = Column(String(255), unique=True, index=True)
#     serial = Column(String(255), unique=True, index=True)
#     purchase_date = Column(Date)
#     purchase_cost = Column(Numeric(10, 2))
#     model_id = Column(Integer, index=True)
#     status_id = Column(Integer, index=True)
#     location_id = Column(Integer, index=True)
#     department_id = Column(Integer, ForeignKey('department.id'), index=True)
#     condition = Column(String(255))
    
#     # Add any other columns you anticipate users asking about
#     department = relationship("Department", back_populates="assets") 
#     location = relationship("Location", back_populates="assets")
#     status = relationship("Status", back_populates="assets")

#     def __repr__(self):
#         return f"<Asset(name='{self.name}', asset_tag='{self.asset_tag}', serial='{self.serial}', purchase_date='{self.purchase_date}')>"

class Asset(Base):
    __tablename__ = 'assets'

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Columns
    assigned_user = Column(Integer, ForeignKey('users.id', ondelete='SET NULL'), nullable=True)
    name = Column(String(255), nullable=True)
    asset_tag = Column(String(255), nullable=False, unique=True, index=True)
    model_id = Column(Integer, ForeignKey('asset_model.id', ondelete='SET NULL'), nullable=True, index=True)
    serial = Column(String(255), nullable=False, unique=True, index=True)
    purchase_date = Column(Date, nullable=True)
    purchase_cost = Column(Numeric(10, 2), nullable=True)
    order_number = Column(String(255), nullable=True)
    manager_id = Column(Integer, ForeignKey('users.id', ondelete='SET NULL'), nullable=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='SET NULL'), nullable=True, index=True)
    description = Column(Text, nullable=True)
    image = Column(String(255), nullable=False, default='images/placeholder.jpg')
    created_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, nullable=True)
    deleted_at = Column(DateTime, nullable=True)
    status_id = Column(Integer, ForeignKey('status.id', ondelete='SET NULL'), nullable=True, index=True)
    archived = Column(Boolean, default=False)
    warrenty_months = Column(Integer, nullable=True)  # tinyint unsigned in MySQL maps to Integer
    warrenty_details = Column(String(255), nullable=True)
    requestable = Column(Boolean, nullable=False, default=True)
    requesting_user = Column(Integer, ForeignKey('users.id', ondelete='SET NULL'), nullable=True)
    location_id = Column(Integer, ForeignKey('location.id', ondelete='SET NULL'), nullable=True, index=True)
    last_moved_date = Column(Date, nullable=True)
    last_checkout = Column(DateTime, nullable=True)
    expected_checkin = Column(Date, nullable=True)
    last_audit_date = Column(DateTime, nullable=True)
    next_audit_date = Column(Date, nullable=True)
    checkin_counter = Column(Integer, nullable=False, default=0)
    checkout_counter = Column(Integer, nullable=False, default=0)
    requests_counter = Column(Integer, nullable=False, default=0)
    manufacturer = Column(String(255), nullable=True)
    current_value = Column(Numeric(10, 2), nullable=False)
    depreciation_method = Column(Integer, ForeignKey('depreciation.id', ondelete='SET NULL'), nullable=True, index=True)
    residual_value = Column(Numeric(10, 2), nullable=True)
    total_cost_ownership = Column(Numeric(20, 2), nullable=True)
    maintenance_cost = Column(Numeric(10, 2), nullable=True)
    condition = Column(String(255), nullable=True)
    utilization_rate = Column(Numeric(10, 2), nullable=False)
    life_cycle_stage = Column(String(255), nullable=True)
    expected_endoflife = Column(Date, nullable=True)
    hazard_information = Column(String(255), nullable=True)
    purchase_from = Column(String(255), nullable=True)
    disposal_date = Column(Date, nullable=True)
    disposal_method = Column(String(255), nullable=True)
    replacement_plan = Column(Boolean, nullable=True)
    replacement_duration = Column(Date, nullable=True)
    replaced_by = Column(Integer, ForeignKey('assets.id', ondelete='SET NULL'), nullable=True, index=True)
    insurance_provider = Column(String(255), nullable=True)
    insurance_policy_number = Column(Integer, nullable=True)
    coverage_amount = Column(Numeric(10, 2), nullable=True)
    insurance_expiry = Column(Date, nullable=True)
    qr_code = Column(Text, nullable=True)
    isCheckedOut = Column(Boolean, nullable=False, default=False)
    department_id = Column(Integer, ForeignKey('department.id', ondelete='CASCADE'), nullable=True, index=True)
    files = Column(Text, nullable=True)  # JSON in MySQL can be mapped to Text in SQLAlchemy
    disposal_reason = Column(String(255), nullable=True)

    # Relationships
    department = relationship("Department", back_populates="assets")
    location = relationship("Location", back_populates="assets")
    status = relationship("Status", back_populates="assets")
    # model = relationship("AssetModel", back_populates="assets")
    manager = relationship("User", foreign_keys=[manager_id], back_populates="managed_assets")
    user = relationship("User", foreign_keys=[user_id], back_populates="owned_assets")
    assigned_to = relationship("User", foreign_keys=[assigned_user], back_populates="assigned_assets")
    requesting = relationship("User", foreign_keys=[requesting_user], back_populates="requested_assets")
    # depreciation = relationship("Depreciation", back_populates="assets")
    replaced_by_asset = relationship("Asset", remote_side=[id], back_populates="replaces")

    # Self-referential relationship for replacement
    replaces = relationship("Asset", back_populates="replaced_by_asset", foreign_keys=[replaced_by])

    def __repr__(self):
        return f"<Asset(name='{self.name}', asset_tag='{self.asset_tag}', serial='{self.serial}')>"