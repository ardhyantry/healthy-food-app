@extends('layouts.adminSB')
@section('title', 'Quickbite - Create Menu')

@section('content')
    <div class="container">
        <h2>Tambah Menu Baru</h2>

        @if(session('success'))
            <div class="alert alert-success">{{ session('success') }}</div>
        @endif

        @if($errors->any())
            <div class="alert alert-danger">
                <ul>
                    @foreach ($errors->all() as $error)
                        <li>{{ $error }}</li>
                    @endforeach
                </ul>
            </div>
        @endif

        <form action="{{ route('admin.menus.store') }}" method="POST">
            @csrf

            <div class="mb-3">
                <label for="name" class="form-label">Nama Menu</label>
                <input type="text" name="name" class="form-control" required>
            </div>

            <div class="mb-3">
                <label for="description" class="form-label">Deskripsi</label>
                <textarea name="description" class="form-control" required></textarea>
            </div>

            <div class="mb-3">
                <label for="nutrition_fact" class="form-label">Fakta Nutrisi</label>
                <textarea name="nutrition_fact" class="form-control"></textarea>
            </div>

            <div class="mb-3">
                <label for="price" class="form-label">Harga</label>
                <input type="number" name="price" class="form-control" required>
            </div>

            <div class="mb-3">
                <label for="stock" class="form-label">Stok</label>
                <input type="number" name="stock" class="form-control" required>
            </div>

            <div class="mb-3">
                <label for="image_path" class="form-label">Path Gambar</label>
                <input type="text" name="image_path" class="form-control" placeholder="contoh: menus/nasi-goreng.jpg" required>
            </div>

            <div class="mb-3">
                <label for="categories_id" class="form-label">Kategori</label>
                <select name="categories_id" class="form-control" required>
                    @foreach($categories as $category)
                        <option value="{{ $category->id }}">{{ $category->name }}</option>
                    @endforeach
                </select>
            </div>

            <button type="submit" class="btn btn-primary">Simpan</button>
            <a href="{{ route('admin.menus.index') }}" class="btn btn-secondary">Kembali</a>
        </form>
    </div>
@endsection
