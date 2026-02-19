# � Healthy Food Ordering Application

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Laravel 10](https://img.shields.io/badge/Laravel-10.x-red.svg)](https://laravel.com)
[![PHP 8.1+](https://img.shields.io/badge/PHP-8.1+-blue.svg)](https://www.php.net)
[![MySQL](https://img.shields.io/badge/MySQL-8.0+-orange.svg)](https://mysql.com)

> **A modern Laravel-based web application for healthy food ordering with admin management and customer ordering features.**

## 🎯 Project Overview

This repository contains a complete **Healthy Food Ordering System** built with Laravel 10. The application provides a seamless experience for customers to browse healthy food options, place orders, and manage their nutrition journey, while providing comprehensive administrative tools for restaurant management.

### Key Features:
- **Customer Portal** - Browse menus, place orders, manage cart and checkout
- **Admin Dashboard** - Manage categories, menus, ingredients, orders, and transactions
- **User Management** - Role-based access control with Laravel Jetstream
- **Order Management** - Complete order lifecycle from cart to delivery
- **Nutrition Tracking** - Ingredient-based meal composition tracking

## 📋 What's Included

### 🏗️ Key Components

#### 1. **Authentication System**
- Laravel Jetstream with Livewire
- Role-based access (Admin/Customer)
- Two-factor authentication support
- User profile management

#### 2. **Menu Management**
- Category-based organization
- Ingredient tracking and composition
- Image upload for food items
- Nutritional information display

#### 3. **Order Processing**
- Shopping cart functionality
- Order management system
- Transaction tracking
- Status updates and notifications

#### 4. **Admin Panel**
- Comprehensive dashboard
- User management
- Order analytics and reporting
- Menu and ingredient management

## 🚀 Quick Start

### Prerequisites
- PHP 8.1 or higher
- Composer
- Node.js & npm
- MySQL database server

### 1. Clone & Install Dependencies
```bash
# Clone the repository
git clone https://github.com/ardhyantry/healthy-food-app.git
cd healthy-food-app

# Install PHP dependencies
composer install

# Install Node.js dependencies  
npm install
```

### 2. Environment Setup
```bash
# Copy environment file
cp .env.example .env

# Generate application key
php artisan key:generate

# Configure your database settings in .env file
DB_DATABASE=laravel
DB_USERNAME=root
DB_PASSWORD=your_password
```

### 3. Database Setup
```bash
# Run migrations to create database tables
php artisan migrate

# Seed the database with sample data (optional)
php artisan db:seed
```

### 4. Build Assets & Run Application
```bash
# Build frontend assets
npm run dev

# Start the development server
php artisan serve
```

Visit `http://localhost:8000` to access the application.

## 🛠️ Technology Stack

### Backend
- **Laravel 10** - PHP framework
- **Laravel Jetstream** - Authentication scaffolding
- **Livewire 3** - Dynamic frontend components
- **MySQL** - Database management

### Frontend
- **Tailwind CSS** - Utility-first CSS framework
- **Alpine.js** - Lightweight JavaScript framework  
- **Blade Templates** - Laravel templating engine
- **Vite** - Frontend build tool

### Development Tools
- **Laravel Sail** - Docker development environment
- **PHPUnit** - Testing framework
- **Laravel Pint** - Code style fixer

## 📊 Database Schema

### Core Tables
- **users** - Customer and admin user accounts
- **categories** - Food category organization
- **menus** - Individual food items
- **ingredients** - Nutritional components
- **menus_has_ingredients** - Menu-ingredient relationships
- **orders** - Customer order items
- **transactions** - Order transaction records

## 🎯 Features Overview

### For Customers
- Browse healthy food categories and menus
- View detailed nutritional information
- Add items to cart and manage quantities
- Secure checkout and payment processing
- Order history and status tracking

### For Administrators  
- Dashboard with sales analytics
- Complete menu and ingredient management
- Order processing and fulfillment
- Customer management and support
- Transaction reporting and insights

## 📈 Future Enhancements

- [ ] Mobile app development
- [ ] Advanced nutrition calculator
- [ ] Meal planning and recommendations
- [ ] Integration with delivery services
- [ ] Customer loyalty program
- [ ] Multi-restaurant support

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

## 📞 Contact & Support

- **Repository**: [ardhyantry/healthy-food-app](https://github.com/ardhyantry/healthy-food-app)
- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Email**: For direct inquiries and support

---

**🍃 Start your healthy eating journey today!**

A modern, feature-rich platform that makes healthy food ordering simple and enjoyable for customers while providing powerful management tools for restaurant operators.
