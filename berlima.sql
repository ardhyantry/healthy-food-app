-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Feb 19, 2026 at 08:14 AM
-- Server version: 10.4.32-MariaDB
-- PHP Version: 8.2.12

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `berlima`
--

-- --------------------------------------------------------

--
-- Table structure for table `categories`
--

CREATE TABLE `categories` (
  `id` bigint(20) UNSIGNED NOT NULL,
  `name` varchar(50) NOT NULL,
  `image_path` varchar(255) NOT NULL,
  `created_at` timestamp NULL DEFAULT NULL,
  `updated_at` timestamp NULL DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

--
-- Dumping data for table `categories`
--

INSERT INTO `categories` (`id`, `name`, `image_path`, `created_at`, `updated_at`) VALUES
(1, 'Appetizer', 'categories/appetizer.jpg', '2025-04-22 04:13:05', '2025-04-22 04:13:05'),
(2, 'Main Course', 'categories/main-course.jpg', '2025-04-22 04:13:05', '2025-04-22 04:13:05'),
(3, 'Snack', 'categories/snack.jpg', '2025-04-22 04:13:05', '2025-04-22 04:13:05'),
(4, 'Dessert', 'categories/dessert.jpg', '2025-04-22 04:13:05', '2025-04-22 04:13:05'),
(5, 'Coffee', 'categories/coffee.jpg', '2025-04-22 04:13:05', '2025-04-22 04:13:05'),
(6, 'Non Coffee', 'categories/non-coffee.jpg', '2025-04-22 04:13:05', '2025-04-22 04:13:05'),
(7, 'Healthy Meals', 'images/categories/healthy-meals.jpg', '2026-02-19 00:04:14', '2026-02-19 00:04:14'),
(8, 'Healthy Beverages', 'images/categories/beverages.jpg', '2026-02-19 00:04:14', '2026-02-19 00:04:14');

-- --------------------------------------------------------

--
-- Table structure for table `failed_jobs`
--

CREATE TABLE `failed_jobs` (
  `id` bigint(20) UNSIGNED NOT NULL,
  `uuid` varchar(255) NOT NULL,
  `connection` text NOT NULL,
  `queue` text NOT NULL,
  `payload` longtext NOT NULL,
  `exception` longtext NOT NULL,
  `failed_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- --------------------------------------------------------

--
-- Table structure for table `ingredients`
--

CREATE TABLE `ingredients` (
  `id` bigint(20) UNSIGNED NOT NULL,
  `name` varchar(50) NOT NULL,
  `created_at` timestamp NULL DEFAULT NULL,
  `updated_at` timestamp NULL DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

--
-- Dumping data for table `ingredients`
--

INSERT INTO `ingredients` (`id`, `name`, `created_at`, `updated_at`) VALUES
(1, 'Daging Sapi', '2025-04-22 04:13:05', '2025-04-22 04:13:05'),
(2, 'Ayam', '2025-04-22 04:13:05', '2025-04-22 04:13:05'),
(3, 'Ikan', '2025-04-22 04:13:05', '2025-04-22 04:13:05'),
(4, 'Nasi', '2025-04-22 04:13:05', '2025-04-22 04:13:05'),
(5, 'Sayuran', '2025-04-22 04:13:05', '2025-04-22 04:13:05'),
(6, 'Bumbu Rahasia', '2025-04-22 04:13:05', '2025-04-22 04:13:05'),
(7, 'Es Batu', '2025-04-22 04:13:05', '2025-04-22 04:13:05'),
(8, 'Gula', '2025-04-22 04:13:05', '2025-04-22 04:13:05'),
(9, 'Susu', '2025-04-22 04:13:05', '2025-04-22 04:13:05'),
(10, 'Coklat', '2025-04-22 04:13:05', '2025-04-22 04:13:05');

-- --------------------------------------------------------

--
-- Table structure for table `menus`
--

CREATE TABLE `menus` (
  `id` bigint(20) UNSIGNED NOT NULL,
  `name` varchar(50) NOT NULL,
  `description` varchar(255) NOT NULL,
  `nutrition_fact` varchar(255) NOT NULL,
  `price` double NOT NULL,
  `stock` int(11) NOT NULL,
  `image_path` varchar(255) NOT NULL,
  `categories_id` bigint(20) UNSIGNED NOT NULL,
  `created_at` timestamp NULL DEFAULT NULL,
  `updated_at` timestamp NULL DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

--
-- Dumping data for table `menus`
--

INSERT INTO `menus` (`id`, `name`, `description`, `nutrition_fact`, `price`, `stock`, `image_path`, `categories_id`, `created_at`, `updated_at`) VALUES
(1, 'Nasi Goreng Spesial', 'Nasi goreng dengan campuran daging dan sayuran', 'Kalori: 500, Protein: 20g', 25000, 50, 'images/menus/nasi-goreng.jpg', 2, '2025-04-22 04:13:05', '2026-02-19 00:09:08'),
(2, 'Es Teh Manis', 'Es teh dengan gula spesial', 'Kalori: 150, Gula: 20g', 8000, 100, 'images/menus/esteh.jpg', 6, '2025-04-22 04:13:05', '2026-02-19 00:09:08'),
(3, 'Kentang Goreng', 'Kentang goreng renyah dengan bumbu spesial', 'Kalori: 300, Lemak: 15g', 15000, 30, 'images/menus/kentang-goreng.png', 4, '2025-04-22 04:13:05', '2026-02-19 00:11:09'),
(5, 'Berry Smoothie', 'Fresh berry smoothie packed with antioxidants', 'Rich in vitamin C, fiber, and antioxidants', 45000, 20, 'images/menus/berry_smoothie.jpg', 8, '2026-02-19 00:04:14', '2026-02-19 00:04:14'),
(6, 'Chia Pudding', 'Creamy chia seed pudding with fresh fruits', 'High in omega-3, fiber, and protein', 35000, 15, 'images/menus/chia_pudding.jpg', 7, '2026-02-19 00:04:14', '2026-02-19 00:04:14'),
(7, 'Chicken Quinoa Bowl', 'Grilled chicken with quinoa and vegetables', 'Complete protein, gluten-free, high fiber', 65000, 12, 'images/menus/chicken_quinoa.jpg', 7, '2026-02-19 00:04:14', '2026-02-19 00:04:14'),
(8, 'Detox Green Juice', 'Fresh green vegetable juice blend', 'Alkalizing, vitamin-rich, low calories', 38000, 25, 'images/menus/detox_juice.jpg', 8, '2026-02-19 00:04:14', '2026-02-19 00:04:14'),
(9, 'Green Tea', 'Premium organic green tea', 'Antioxidant-rich, metabolism boosting', 25000, 30, 'images/menus/green_tea.webp', 8, '2026-02-19 00:04:14', '2026-02-19 00:04:14'),
(10, 'Kale Caesar Salad', 'Fresh kale salad with healthy caesar dressing', 'Iron-rich, vitamin K, low carb', 42000, 18, 'images/menus/kale_salad.jpg', 7, '2026-02-19 00:04:14', '2026-02-19 00:04:14'),
(11, 'Spinach Veggie Wrap', 'Whole wheat wrap with fresh spinach and vegetables', 'Fiber-rich, vitamin-packed, balanced meal', 48000, 14, 'images/menus/spinach_wrap.jpg', 7, '2026-02-19 00:04:14', '2026-02-19 00:04:14'),
(12, 'Sweet Potato Soup', 'Creamy roasted sweet potato soup', 'Beta-carotene, vitamin A, comfort food', 36000, 20, 'images/menus/sweet_potato_soup.webp', 7, '2026-02-19 00:04:14', '2026-02-19 00:04:14'),
(13, 'Rainbow Veggie Bowl', 'Colorful bowl of seasonal vegetables', 'Multivitamin natural source, high fiber', 52000, 16, 'images/menus/veggie_bowl.jpg', 7, '2026-02-19 00:04:14', '2026-02-19 00:04:14'),
(14, 'Mediterranean Veggie Platter', 'Mediterranean-style vegetable arrangement', 'Heart-healthy, Mediterranean diet', 58000, 10, 'images/menus/veggie_platter.jpg', 7, '2026-02-19 00:04:14', '2026-02-19 00:04:14');

-- --------------------------------------------------------

--
-- Table structure for table `menus_has_ingredients`
--

CREATE TABLE `menus_has_ingredients` (
  `id` bigint(20) UNSIGNED NOT NULL,
  `menus_id` bigint(20) UNSIGNED NOT NULL,
  `ingredients_id` bigint(20) UNSIGNED NOT NULL,
  `created_at` timestamp NULL DEFAULT NULL,
  `updated_at` timestamp NULL DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

--
-- Dumping data for table `menus_has_ingredients`
--

INSERT INTO `menus_has_ingredients` (`id`, `menus_id`, `ingredients_id`, `created_at`, `updated_at`) VALUES
(1, 1, 1, '2025-04-22 04:13:05', '2025-04-22 04:13:05'),
(2, 1, 4, '2025-04-22 04:13:05', '2025-04-22 04:13:05'),
(3, 1, 5, '2025-04-22 04:13:05', '2025-04-22 04:13:05'),
(4, 2, 7, '2025-04-22 04:13:05', '2025-04-22 04:13:05'),
(5, 2, 8, '2025-04-22 04:13:05', '2025-04-22 04:13:05'),
(6, 3, 5, '2025-04-22 04:13:05', '2025-04-22 04:13:05');

-- --------------------------------------------------------

--
-- Table structure for table `migrations`
--

CREATE TABLE `migrations` (
  `id` int(10) UNSIGNED NOT NULL,
  `migration` varchar(255) NOT NULL,
  `batch` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

--
-- Dumping data for table `migrations`
--

INSERT INTO `migrations` (`id`, `migration`, `batch`) VALUES
(1, '2014_10_12_100000_create_password_reset_tokens_table', 1),
(2, '2019_08_19_000000_create_failed_jobs_table', 1),
(3, '2019_12_14_000001_create_personal_access_tokens_table', 1),
(4, '2025_04_20_115022_create_users_table', 1),
(5, '2025_04_20_115023_create_categories_table', 1),
(6, '2025_04_20_115026_create_menus_table', 1),
(7, '2025_04_20_115028_create_ingredients_table', 1),
(8, '2025_04_20_115030_create_menus_has_ingredients_table', 1),
(9, '2025_04_20_115032_create_transactions_table', 1),
(10, '2025_04_21_113050_update_status_in_transactions_table', 1),
(11, '2025_04_22_010813_create_orders_table', 1),
(12, '2014_10_12_200000_add_two_factor_columns_to_users_table', 2),
(13, '2025_06_25_065316_create_sessions_table', 2);

-- --------------------------------------------------------

--
-- Table structure for table `orders`
--

CREATE TABLE `orders` (
  `id` bigint(20) UNSIGNED NOT NULL,
  `transactions_id` bigint(20) UNSIGNED NOT NULL,
  `menus_id` bigint(20) UNSIGNED NOT NULL,
  `portion_size` enum('small','medium','large') DEFAULT NULL,
  `quantity` int(11) NOT NULL DEFAULT 1,
  `total` double NOT NULL DEFAULT 0,
  `notes` text DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT NULL,
  `updated_at` timestamp NULL DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

--
-- Dumping data for table `orders`
--

INSERT INTO `orders` (`id`, `transactions_id`, `menus_id`, `portion_size`, `quantity`, `total`, `notes`, `created_at`, `updated_at`) VALUES
(1, 1, 1, 'medium', 2, 50000, 'Tanpa sambal', '2025-04-22 04:13:05', '2025-04-22 04:13:05'),
(2, 1, 3, 'large', 1, 15000, NULL, '2025-04-22 04:13:05', '2025-04-22 04:13:05'),
(3, 2, 2, 'small', 3, 24000, 'Gula sedikit', '2025-04-22 04:13:05', '2025-04-22 04:13:05'),
(5, 4, 1, 'large', 2, 50000, NULL, '2025-04-22 04:36:20', '2025-04-22 04:36:48');

-- --------------------------------------------------------

--
-- Table structure for table `password_reset_tokens`
--

CREATE TABLE `password_reset_tokens` (
  `email` varchar(255) NOT NULL,
  `token` varchar(255) NOT NULL,
  `created_at` timestamp NULL DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- --------------------------------------------------------

--
-- Table structure for table `personal_access_tokens`
--

CREATE TABLE `personal_access_tokens` (
  `id` bigint(20) UNSIGNED NOT NULL,
  `tokenable_type` varchar(255) NOT NULL,
  `tokenable_id` bigint(20) UNSIGNED NOT NULL,
  `name` varchar(255) NOT NULL,
  `token` varchar(64) NOT NULL,
  `abilities` text DEFAULT NULL,
  `last_used_at` timestamp NULL DEFAULT NULL,
  `expires_at` timestamp NULL DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT NULL,
  `updated_at` timestamp NULL DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- --------------------------------------------------------

--
-- Table structure for table `sessions`
--

CREATE TABLE `sessions` (
  `id` varchar(255) NOT NULL,
  `user_id` bigint(20) UNSIGNED DEFAULT NULL,
  `ip_address` varchar(45) DEFAULT NULL,
  `user_agent` text DEFAULT NULL,
  `payload` longtext NOT NULL,
  `last_activity` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

--
-- Dumping data for table `sessions`
--

INSERT INTO `sessions` (`id`, `user_id`, `ip_address`, `user_agent`, `payload`, `last_activity`) VALUES
('8c9uyb2KceL298ITnFd6v58iZ16I9QFi7wGVHUKU', NULL, '127.0.0.1', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Code/1.109.3 Chrome/142.0.7444.265 Electron/39.3.0 Safari/537.36', 'YTozOntzOjY6Il90b2tlbiI7czo0MDoieWx5a0oyOG1UcjN5bkhYWGQ4aVA4ZWV3WFpDTzZiU1dXMXlyOElUciI7czo5OiJfcHJldmlvdXMiO2E6MTp7czozOiJ1cmwiO3M6MzM6Imh0dHA6Ly9sb2NhbGhvc3Q6ODAwMC9kZWJ1Zy1tZW51cyI7fXM6NjoiX2ZsYXNoIjthOjI6e3M6Mzoib2xkIjthOjA6e31zOjM6Im5ldyI7YTowOnt9fX0=', 1771484876),
('A19xikawYXbbIWqDfvaZTbXjI0xSt68itRe9nRlJ', NULL, '127.0.0.1', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Code/1.109.3 Chrome/142.0.7444.265 Electron/39.3.0 Safari/537.36', 'YTozOntzOjY6Il90b2tlbiI7czo0MDoiNzc0dENZZ0NTbzRwbEhiZlNHYzdQYmNMWUJYeDZxS2NqOGl2RHdlYSI7czo5OiJfcHJldmlvdXMiO2E6MTp7czozOiJ1cmwiO3M6MzM6Imh0dHA6Ly9sb2NhbGhvc3Q6ODAwMC9kZWJ1Zy1tZW51cyI7fXM6NjoiX2ZsYXNoIjthOjI6e3M6Mzoib2xkIjthOjA6e31zOjM6Im5ldyI7YTowOnt9fX0=', 1771484871),
('vD16GeO6xVp8ZhWL0mWIKMXkC9rwvkxFptQmSeth', NULL, '127.0.0.1', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36', 'YTo0OntzOjY6Il90b2tlbiI7czo0MDoieGNOUE1hbFNPVmowTm9WQ1ZqNm9aTmtwNzdvQ3hkWFMyOWppRjd2YyI7czo5OiJfcHJldmlvdXMiO2E6MTp7czozOiJ1cmwiO3M6MjY6Imh0dHA6Ly8xMjcuMC4wLjE6ODAwMC9ob21lIjt9czo2OiJfZmxhc2giO2E6Mjp7czozOiJvbGQiO2E6MDp7fXM6MzoibmV3IjthOjA6e319czozOiJ1cmwiO2E6MTp7czo4OiJpbnRlbmRlZCI7czoyNzoiaHR0cDovLzEyNy4wLjAuMTo4MDAwL21lbnVzIjt9fQ==', 1771485263);

-- --------------------------------------------------------

--
-- Table structure for table `transactions`
--

CREATE TABLE `transactions` (
  `id` bigint(20) UNSIGNED NOT NULL,
  `subtotal` double NOT NULL,
  `discount` double NOT NULL DEFAULT 0,
  `total` double NOT NULL,
  `order_type` enum('dine_in','take_away') NOT NULL,
  `payment_type` enum('QRIS','credit_card','debit_card','e_wallet') NOT NULL,
  `status` enum('pending','processing','ready','cancelled') NOT NULL DEFAULT 'pending',
  `users_id` bigint(20) UNSIGNED NOT NULL,
  `created_at` timestamp NULL DEFAULT NULL,
  `updated_at` timestamp NULL DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

--
-- Dumping data for table `transactions`
--

INSERT INTO `transactions` (`id`, `subtotal`, `discount`, `total`, `order_type`, `payment_type`, `status`, `users_id`, `created_at`, `updated_at`) VALUES
(1, 33000, 0, 33000, 'take_away', 'QRIS', 'pending', 2, '2025-04-20 04:13:05', '2025-04-22 04:22:06'),
(2, 40000, 5000, 35000, 'dine_in', 'e_wallet', 'processing', 2, '2025-04-21 04:13:05', '2025-04-21 04:13:05'),
(4, 1, 1, 1, 'dine_in', 'QRIS', 'processing', 3, '2025-04-22 04:35:50', '2025-04-22 04:35:50');

-- --------------------------------------------------------

--
-- Table structure for table `users`
--

CREATE TABLE `users` (
  `id` bigint(20) UNSIGNED NOT NULL,
  `name` varchar(50) NOT NULL,
  `email` varchar(50) NOT NULL,
  `phone_number` varchar(25) NOT NULL,
  `username` varchar(45) NOT NULL,
  `password` varchar(255) NOT NULL,
  `two_factor_secret` text DEFAULT NULL,
  `two_factor_recovery_codes` text DEFAULT NULL,
  `two_factor_confirmed_at` timestamp NULL DEFAULT NULL,
  `role` enum('admin','customer') NOT NULL DEFAULT 'customer',
  `created_at` timestamp NULL DEFAULT NULL,
  `updated_at` timestamp NULL DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

--
-- Dumping data for table `users`
--

INSERT INTO `users` (`id`, `name`, `email`, `phone_number`, `username`, `password`, `two_factor_secret`, `two_factor_recovery_codes`, `two_factor_confirmed_at`, `role`, `created_at`, `updated_at`) VALUES
(1, 'Admin', 'admin@resto.com', '081234567890', 'admin', '$2y$10$o8REUAbmSnQyDuYOIexfnO3fIxbKF500yJB3oDvydFv94powlQArm', NULL, NULL, NULL, 'admin', '2025-04-22 04:13:05', '2025-04-22 04:13:05'),
(2, 'Customer', 'customer@resto.com', '081234567891', 'customer', '$2y$10$kLLGu2nC54ThbdTicX1m3ujW2fy0BpSWZTlnUXTK.v/FVey0qYQy2', NULL, NULL, NULL, 'customer', '2025-04-22 04:13:05', '2025-04-22 04:13:05'),
(3, 'Runner up', '21@gmai.com', '2123', 's160422083', '$2y$10$EEAkvKdZkaFZ2DVi9Ln4Je1UHiWyGjQlNIumGJjKAoZ/9ofFdSnRW', NULL, NULL, NULL, 'customer', '2025-04-22 04:34:16', '2025-04-22 04:34:16'),
(4, 'John Doe', 'john.doe@example.com', '081234567892', 'johndoe', '$2y$10$92IXUNpkjO0rOQ5byMi.Ye4oKoEa3Ro9llC/.og/at2.uheWG/igi', NULL, NULL, NULL, 'customer', '2026-02-19 07:30:00', '2026-02-19 07:30:00'),
(5, 'Jane Smith', 'jane.smith@example.com', '081234567893', 'janesmith', '$2y$10$92IXUNpkjO0rOQ5byMi.Ye4oKoEa3Ro9llC/.og/at2.uheWG/igi', NULL, NULL, NULL, 'customer', '2026-02-19 07:30:00', '2026-02-19 07:30:00');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `categories`
--
ALTER TABLE `categories`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `failed_jobs`
--
ALTER TABLE `failed_jobs`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `failed_jobs_uuid_unique` (`uuid`);

--
-- Indexes for table `ingredients`
--
ALTER TABLE `ingredients`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `menus`
--
ALTER TABLE `menus`
  ADD PRIMARY KEY (`id`),
  ADD KEY `menus_categories_id_foreign` (`categories_id`);

--
-- Indexes for table `menus_has_ingredients`
--
ALTER TABLE `menus_has_ingredients`
  ADD PRIMARY KEY (`id`),
  ADD KEY `menus_has_ingredients_menus_id_foreign` (`menus_id`),
  ADD KEY `menus_has_ingredients_ingredients_id_foreign` (`ingredients_id`);

--
-- Indexes for table `migrations`
--
ALTER TABLE `migrations`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `orders`
--
ALTER TABLE `orders`
  ADD PRIMARY KEY (`id`),
  ADD KEY `orders_transactions_id_foreign` (`transactions_id`),
  ADD KEY `orders_menus_id_foreign` (`menus_id`);

--
-- Indexes for table `password_reset_tokens`
--
ALTER TABLE `password_reset_tokens`
  ADD PRIMARY KEY (`email`);

--
-- Indexes for table `personal_access_tokens`
--
ALTER TABLE `personal_access_tokens`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `personal_access_tokens_token_unique` (`token`),
  ADD KEY `personal_access_tokens_tokenable_type_tokenable_id_index` (`tokenable_type`,`tokenable_id`);

--
-- Indexes for table `sessions`
--
ALTER TABLE `sessions`
  ADD PRIMARY KEY (`id`),
  ADD KEY `sessions_user_id_index` (`user_id`),
  ADD KEY `sessions_last_activity_index` (`last_activity`);

--
-- Indexes for table `transactions`
--
ALTER TABLE `transactions`
  ADD PRIMARY KEY (`id`),
  ADD KEY `transactions_users_id_foreign` (`users_id`);

--
-- Indexes for table `users`
--
ALTER TABLE `users`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `users_email_unique` (`email`),
  ADD UNIQUE KEY `users_username_unique` (`username`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `categories`
--
ALTER TABLE `categories`
  MODIFY `id` bigint(20) UNSIGNED NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=9;

--
-- AUTO_INCREMENT for table `failed_jobs`
--
ALTER TABLE `failed_jobs`
  MODIFY `id` bigint(20) UNSIGNED NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `ingredients`
--
ALTER TABLE `ingredients`
  MODIFY `id` bigint(20) UNSIGNED NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=11;

--
-- AUTO_INCREMENT for table `menus`
--
ALTER TABLE `menus`
  MODIFY `id` bigint(20) UNSIGNED NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=15;

--
-- AUTO_INCREMENT for table `menus_has_ingredients`
--
ALTER TABLE `menus_has_ingredients`
  MODIFY `id` bigint(20) UNSIGNED NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=7;

--
-- AUTO_INCREMENT for table `migrations`
--
ALTER TABLE `migrations`
  MODIFY `id` int(10) UNSIGNED NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=14;

--
-- AUTO_INCREMENT for table `orders`
--
ALTER TABLE `orders`
  MODIFY `id` bigint(20) UNSIGNED NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=6;

--
-- AUTO_INCREMENT for table `personal_access_tokens`
--
ALTER TABLE `personal_access_tokens`
  MODIFY `id` bigint(20) UNSIGNED NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `transactions`
--
ALTER TABLE `transactions`
  MODIFY `id` bigint(20) UNSIGNED NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=5;

--
-- AUTO_INCREMENT for table `users`
--
ALTER TABLE `users`
  MODIFY `id` bigint(20) UNSIGNED NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=6;

--
-- Constraints for dumped tables
--

--
-- Constraints for table `menus`
--
ALTER TABLE `menus`
  ADD CONSTRAINT `menus_categories_id_foreign` FOREIGN KEY (`categories_id`) REFERENCES `categories` (`id`);

--
-- Constraints for table `menus_has_ingredients`
--
ALTER TABLE `menus_has_ingredients`
  ADD CONSTRAINT `menus_has_ingredients_ingredients_id_foreign` FOREIGN KEY (`ingredients_id`) REFERENCES `ingredients` (`id`),
  ADD CONSTRAINT `menus_has_ingredients_menus_id_foreign` FOREIGN KEY (`menus_id`) REFERENCES `menus` (`id`);

--
-- Constraints for table `orders`
--
ALTER TABLE `orders`
  ADD CONSTRAINT `orders_menus_id_foreign` FOREIGN KEY (`menus_id`) REFERENCES `menus` (`id`) ON DELETE CASCADE,
  ADD CONSTRAINT `orders_transactions_id_foreign` FOREIGN KEY (`transactions_id`) REFERENCES `transactions` (`id`) ON DELETE CASCADE;

--
-- Constraints for table `transactions`
--
ALTER TABLE `transactions`
  ADD CONSTRAINT `transactions_users_id_foreign` FOREIGN KEY (`users_id`) REFERENCES `users` (`id`);
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
